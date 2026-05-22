from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
import torch.nn.functional as F

import stable_worldmodel as swm


def stablewm_cache_dir(sub_folder: str | None = None) -> Path:
    try:
        return Path(swm.data.utils.get_cache_dir(sub_folder=sub_folder))
    except TypeError:
        root = Path(swm.data.utils.get_cache_dir())
        if sub_folder == "checkpoints":
            return root
        if sub_folder is None:
            return root
        return root / sub_folder


def _resolve_checkpoint(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    if path.exists():
        return path

    base = stablewm_cache_dir(sub_folder="checkpoints")
    run_path = base / str(path_or_name)
    if run_path.exists() and run_path.is_file():
        return run_path
    pt_path = Path(f"{run_path}.pt")
    if pt_path.exists():
        return pt_path
    if run_path.is_dir():
        ckpts = sorted(
            list(run_path.glob("*.pt")) + list(run_path.glob("*_object.ckpt")),
            key=lambda x: x.stat().st_ctime,
            reverse=True,
        )
        if ckpts:
            return ckpts[0]

    object_path = Path(f"{run_path}_object.ckpt")
    if object_path.exists():
        return object_path

    raise FileNotFoundError(f"Could not resolve checkpoint: {path_or_name}")


def load_lewm(lewm_checkpoint: str | Path) -> nn.Module:
    path = Path(lewm_checkpoint)
    if not path.exists() and not path.is_absolute():
        cache_path = stablewm_cache_dir() / path
        if cache_path.exists():
            path = cache_path
    if path.exists() and path.is_file():
        module = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(module, (dict, OrderedDict)):
            state_dict = module.get("state_dict", module)
            return _build_lewm_from_state_dict(state_dict).eval()

        def scan(child):
            if hasattr(child, "get_cost"):
                return child.eval() if isinstance(child, nn.Module) else child
            if isinstance(child, nn.Module):
                for grandchild in child.children():
                    found = scan(grandchild)
                    if found is not None:
                        return found
            return None

        found = scan(module)
        if found is None:
            raise RuntimeError(f"No module with get_cost found in {lewm_checkpoint}")
        return found

    try:
        return swm.policy.AutoCostModel(str(lewm_checkpoint)).eval()
    except AttributeError:
        path = _resolve_checkpoint(lewm_checkpoint)
        module = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(module, (dict, OrderedDict)):
            state_dict = module.get("state_dict", module)
            return _build_lewm_from_state_dict(state_dict).eval()
        raise


def _build_lewm_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    import stable_pretraining as spt

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    state_dict = OrderedDict(
        (k.removeprefix("model."), v) for k, v in state_dict.items()
    )
    hidden_dim = state_dict["encoder.embeddings.cls_token"].shape[-1]
    patch_size = state_dict["encoder.embeddings.patch_embeddings.projection.weight"].shape[-1]
    num_patches = state_dict["encoder.embeddings.position_embeddings"].shape[1] - 1
    image_size = int(num_patches**0.5) * patch_size
    embed_dim = state_dict["projector.net.3.bias"].shape[0]
    projector_hidden_dim = state_dict["projector.net.0.bias"].shape[0]
    action_dim = state_dict["action_encoder.patch_embed.weight"].shape[1]
    smoothed_dim = state_dict["action_encoder.patch_embed.weight"].shape[0]
    action_mlp_hidden = state_dict["action_encoder.embed.0.bias"].shape[0]
    action_mlp_scale = max(action_mlp_hidden // embed_dim, 1)
    num_frames = state_dict["predictor.pos_embedding"].shape[1]
    predictor_depth = max(
        int(k.split(".")[3])
        for k in state_dict
        if k.startswith("predictor.transformer.layers.")
    ) + 1
    mlp_dim = state_dict["predictor.transformer.layers.0.mlp.net.1.bias"].shape[0]
    inner_dim = state_dict["predictor.transformer.layers.0.attn.to_out.0.weight"].shape[1]
    dim_head = 64 if inner_dim % 64 == 0 else hidden_dim
    heads = inner_dim // dim_head
    encoder_scale = {
        192: "tiny",
        384: "small",
        768: "base",
        1024: "large",
    }.get(hidden_dim)
    if encoder_scale is None:
        raise ValueError(f"Cannot infer ViT scale from hidden_dim={hidden_dim}")

    encoder = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=image_size,
        pretrained=False,
        use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=num_frames,
        depth=predictor_depth,
        heads=heads,
        mlp_dim=mlp_dim,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dim_head=dim_head,
        dropout=0.0,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=smoothed_dim,
        emb_dim=embed_dim,
        mlp_scale=action_mlp_scale,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=projector_hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=projector_hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.load_state_dict(state_dict, strict=True)
    return model


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentPathFlow(nn.Module):
    """Rectified-flow velocity model for latent path interiors."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        max_horizon: int = 20,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.depth = depth
        self.time_dim = time_dim
        self.dropout = dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon - 1, hidden_dim) * 0.02
        )
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.start_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens = x_t.size(1)
        if n_tokens > self.max_horizon - 1:
            raise ValueError(
                f"Requested {n_tokens + 1} horizon, but max_horizon={self.max_horizon}"
            )
        cond = (
            self.start_proj(z_start)
            + self.goal_proj(z_goal)
            + self.time_embed(t.float())
        )
        x = self.token_proj(x_t)
        x = x + self.pos_embedding[:, :n_tokens] + cond[:, None]
        return self.out(self.net(x))


class InverseDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        layers: list[nn.Module] = []
        in_dim = latent_dim * 3
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, z_next, z_next - z_t], dim=-1)
        return self.net(x)


class LatentPlannerRuntime(nn.Module):
    def __init__(
        self,
        lewm: nn.Module,
        flow: LatentPathFlow,
        inverse_dynamics: InverseDynamics,
        action_block: int = 5,
    ):
        super().__init__()
        self.lewm = lewm.eval()
        self.flow = flow
        self.inverse_dynamics = inverse_dynamics
        self.action_block = action_block
        self.rollout_count = 0
        self.lewm.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str | torch.device = "cpu"):
        path = _resolve_checkpoint(checkpoint)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(
                "Latent planner checkpoints must be dicts with state_dict entries, "
                "not serialized model objects."
            )

        arch = payload["arch"]
        lewm = load_lewm(payload["lewm_checkpoint"])
        flow = LatentPathFlow(**arch["flow"])
        inverse_dynamics = InverseDynamics(**arch["inverse_dynamics"])
        flow.load_state_dict(payload["flow_state_dict"])
        inverse_dynamics.load_state_dict(payload["inverse_dynamics_state_dict"])
        model = cls(
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            action_block=payload.get("action_block", 1),
        )
        return model.to(device).eval()

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        info = {"pixels": pixels.to(self.device)}
        return self.lewm.encode(info)["emb"]

    @torch.no_grad()
    def encode_current_and_goal(self, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = info_dict["pixels"].to(self.device)
        goal = info_dict["goal"].to(self.device)
        z_start = self.encode_pixels(pixels)[:, -1]
        z_goal = self.encode_pixels(goal)[:, -1]
        return z_start, z_goal

    @torch.no_grad()
    def sample_paths(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        b, d = z_start.shape
        n = b * num_samples
        z0 = z_start[:, None].expand(b, num_samples, d).reshape(n, d)
        zg = z_goal[:, None].expand(b, num_samples, d).reshape(n, d)
        x = torch.randn(
            n,
            horizon - 1,
            d,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )
        dt = 1.0 / max(flow_steps, 1)
        for i in range(flow_steps):
            t = torch.full((n,), i * dt, device=z_start.device, dtype=z_start.dtype)
            x = x + self.flow(x, t, z0, zg) * dt
        start = z0[:, None]
        goal = zg[:, None]
        return torch.cat([start, x, goal], dim=1).reshape(b, num_samples, horizon + 1, d)

    def decode_actions(self, paths: torch.Tensor) -> torch.Tensor:
        z_t = paths[..., :-1, :]
        z_next = paths[..., 1:, :]
        flat_actions = self.inverse_dynamics(
            z_t.reshape(-1, z_t.size(-1)),
            z_next.reshape(-1, z_next.size(-1)),
        )
        return flat_actions.reshape(*z_t.shape[:-1], -1)

    @torch.no_grad()
    def rollout_final_latent(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        b, s, h = actions.shape[:3]
        z = z_start[:, None, None].expand(b, s, 1, -1).reshape(b * s, 1, -1).clone()
        act = actions.reshape(b * s, h, -1).to(self.device)
        for t in range(h):
            act_hist = act[:, max(0, t - history_size + 1) : t + 1]
            emb_hist = z[:, -act_hist.size(1) :]
            act_emb = self.lewm.action_encoder(act_hist)
            pred = self.lewm.predict(emb_hist, act_emb)[:, -1:]
            z = torch.cat([z, pred], dim=1)
        self.rollout_count += b * s
        return z[:, -1].reshape(b, s, -1)

    @torch.no_grad()
    def plan(
        self,
        info_dict: dict,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        history_size: int = 3,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        z_start, z_goal = self.encode_current_and_goal(info_dict)
        paths = self.sample_paths(
            z_start,
            z_goal,
            horizon=horizon,
            num_samples=num_samples,
            flow_steps=flow_steps,
            generator=generator,
        )
        actions = self.decode_actions(paths)
        score_mode = score_mode.lower()
        if score_mode == "rollout_goal":
            rollout_final = self.rollout_final_latent(z_start, actions, history_size)
            goal_cost = F.mse_loss(
                rollout_final,
                z_goal[:, None].expand_as(rollout_final),
                reduction="none",
            ).mean(dim=-1)
            cost = goal_weight * goal_cost
        elif score_mode == "first":
            # No reranking ablation: execute the first sampled candidate as-is.
            goal_cost = torch.zeros(actions.shape[:2], device=actions.device, dtype=actions.dtype)
            cost = goal_cost.clone()
        elif score_mode == "path_smoothness":
            goal_cost = torch.zeros(actions.shape[:2], device=actions.device, dtype=actions.dtype)
            accel = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            cost = accel.pow(2).mean(dim=(-1, -2))
        else:
            raise ValueError(
                f"Unknown latent planner score_mode={score_mode!r}. "
                "Expected one of: rollout_goal, first, path_smoothness."
            )

        if consistency_weight:
            rollout_paths = self.rollout_paths(z_start, actions, history_size)
            consistency = (rollout_paths - paths).pow(2).mean(dim=(-1, -2))
            cost = cost + consistency_weight * consistency

        if smoothness_weight:
            accel = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            smooth = accel.pow(2).mean(dim=(-1, -2))
            cost = cost + smoothness_weight * smooth

        best = cost.argmin(dim=1)
        batch_idx = torch.arange(actions.size(0), device=actions.device)
        return {
            "actions": actions[batch_idx, best].detach().cpu(),
            "costs": cost[batch_idx, best].detach().cpu(),
            "all_costs": cost.detach().cpu(),
            "goal_costs": goal_cost.detach().cpu(),
        }

    @torch.no_grad()
    def rollout_paths(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        b, s, h = actions.shape[:3]
        z = z_start[:, None, None].expand(b, s, 1, -1).reshape(b * s, 1, -1).clone()
        act = actions.reshape(b * s, h, -1).to(self.device)
        for t in range(h):
            act_hist = act[:, max(0, t - history_size + 1) : t + 1]
            emb_hist = z[:, -act_hist.size(1) :]
            act_emb = self.lewm.action_encoder(act_hist)
            pred = self.lewm.predict(emb_hist, act_emb)[:, -1:]
            z = torch.cat([z, pred], dim=1)
        return z.reshape(b, s, h + 1, -1)


class LearnedLatentPathSolver:
    """stable-worldmodel solver wrapper for the learned latent planner."""

    def __init__(
        self,
        checkpoint: str | Path,
        batch_size: int = 1,
        num_samples: int = 64,
        flow_steps: int = 16,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        history_size: int = 3,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.score_mode = score_mode
        self.goal_weight = goal_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.history_size = history_size
        self.model = LatentPlannerRuntime.from_checkpoint(checkpoint, device=self.device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        if not isinstance(action_space, Box):
            raise TypeError(f"LearnedLatentPathSolver expects Box action space, got {type(action_space)}")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        all_actions = []
        all_costs = []
        all_goal_costs = []
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch = {k: v[start:end] for k, v in info_dict.items()}
            out = self.model.plan(
                batch,
                horizon=self.horizon,
                num_samples=self.num_samples,
                flow_steps=self.flow_steps,
                score_mode=self.score_mode,
                goal_weight=self.goal_weight,
                consistency_weight=self.consistency_weight,
                smoothness_weight=self.smoothness_weight,
                history_size=self.history_size,
                generator=self.torch_gen,
            )
            all_actions.append(out["actions"])
            all_costs.append(out["costs"])
            all_goal_costs.append(out["goal_costs"])
        return {
            "actions": torch.cat(all_actions, dim=0),
            "costs": torch.cat(all_costs, dim=0).tolist(),
            "goal_costs": torch.cat(all_goal_costs, dim=0),
            "rollout_count": self.model.rollout_count,
        }


def flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    pred_v = flow(x_t, t, z_start, z_goal)
    return F.mse_loss(pred_v, target - noise)


def inverse_dynamics_loss(
    inverse_dynamics: InverseDynamics,
    z_path: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = inverse_dynamics(z_path[:, :-1].reshape(-1, z_path.size(-1)), z_path[:, 1:].reshape(-1, z_path.size(-1)))
    pred = pred.reshape(z_path.size(0), z_path.size(1) - 1, -1)
    return F.mse_loss(pred, actions), pred


def lewm_consistency_loss(
    lewm: nn.Module,
    z_path: torch.Tensor,
    pred_actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    losses = []
    for t in range(pred_actions.size(1)):
        start = max(0, t - history_size + 1)
        emb_hist = z_path[:, start : t + 1]
        act_hist = pred_actions[:, start : t + 1]
        act_emb = lewm.action_encoder(act_hist)
        pred = lewm.predict(emb_hist, act_emb)[:, -1]
        losses.append(F.mse_loss(pred, z_path[:, t + 1]))
    return torch.stack(losses).mean()


def smoothness_loss(z_path: torch.Tensor) -> torch.Tensor:
    if z_path.size(1) < 3:
        return z_path.new_tensor(0.0)
    accel = z_path[:, 2:] - 2 * z_path[:, 1:-1] + z_path[:, :-2]
    return accel.pow(2).mean()


def checkpoint_payload(
    *,
    lewm_checkpoint: str,
    action_block: int,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lewm_checkpoint": lewm_checkpoint,
        "action_block": action_block,
        "arch": {
            "flow": {
                "latent_dim": flow.latent_dim,
                "hidden_dim": flow.hidden_dim,
                "depth": flow.depth,
                "max_horizon": flow.max_horizon,
                "time_dim": flow.time_dim,
                "dropout": flow.dropout,
            },
            "inverse_dynamics": {
                "latent_dim": inverse_dynamics.latent_dim,
                "action_dim": inverse_dynamics.action_dim,
                "hidden_dim": inverse_dynamics.hidden_dim,
                "depth": inverse_dynamics.depth,
                "dropout": inverse_dynamics.dropout,
            },
        },
        "config": cfg,
        "flow_state_dict": flow.state_dict(),
        "inverse_dynamics_state_dict": inverse_dynamics.state_dict(),
    }
