# LeFlow: Latent Path Flow Planning for LeWorldModel

LeFlow adds a learned goal-conditioned planner on top of a frozen
[LeWorldModel](https://arxiv.org/pdf/2603.19312v1) backbone. Instead of running
test-time CEM over actions, LeFlow first generates a latent future path and then
uses inverse dynamics to convert latent transitions into action chunks.

This repository builds on the original LeWorldModel codebase, plus
`stable-worldmodel` for datasets, environments, policies, and evaluation.

## Checkpoints

Main H=5 LeFlow checkpoints are hosted on [`Hugging Face`](https://huggingface.co/hsiangwei0903/LeFlow)

The repo contains:

| Benchmark | Checkpoint |
|---|---|
| TwoRoom | `tworoom/latent_planner.pt` |
| PushT | `pusht/latent_planner.pt` |
| Reacher | `reacher/latent_planner.pt` |
| OGBench Cube | `cube/latent_planner.pt` |

The LeFlow checkpoint stores only lightweight planner state:

- flow planner state dict
- inverse dynamics state dict
- architecture metadata
- reference to the frozen LeWM checkpoint


## Setup

Create an environment following the original LeWorldModel setup:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Set the stable-worldmodel cache root:

```bash
export STABLEWM_HOME=/path/to/stable-wm
```

You also need the original LeWM datasets and frozen LeWM checkpoints from the
official LeWM collection:

[`quentinll/lewm`](https://huggingface.co/collections/quentinll/lewm)

Expected frozen LeWM checkpoint references:

| Benchmark | LeWM reference |
|---|---|
| TwoRoom | `tworoom/lewm` |
| PushT | `pusht/lewm` |
| Reacher | `reacher/lewm` |
| OGBench Cube | `cube/lewm` |

These should resolve under `$STABLEWM_HOME`, exactly as in the original LeWM
evaluation code.

## Evaluation

Use `solver=latent_flow` and pass the downloaded LeFlow checkpoint as `policy`.

TwoRoom:

```bash
python eval.py --config-name=tworoom.yaml \
  solver=latent_flow \
  policy=leflow/tworoom/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

PushT:

```bash
python eval.py --config-name=pusht.yaml \
  solver=latent_flow \
  policy=leflow/pusht/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

Reacher:

```bash
python eval.py --config-name=reacher.yaml \
  solver=latent_flow \
  policy=leflow/reacher/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

OGBench Cube:

```bash
python eval.py --config-name=cube.yaml \
  solver=latent_flow \
  policy=leflow/cube/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

Default latent-flow inference settings are in
[`config/eval/solver/latent_flow.yaml`](config/eval/solver/latent_flow.yaml):

```yaml
num_samples: 64
flow_steps: 16
score_mode: rollout_goal
```

The reranking score uses frozen-LeWM autoregressive rollout final latent
distance to the encoded goal. It does not score candidates by the clamped
generated endpoint.

## Training

LeFlow training freezes LeWM and trains two modules jointly:

- `LatentPathFlow`: rectified-flow model over latent path interiors
- `InverseDynamics`: maps `[z_t, z_{t+1}, z_{t+1} - z_t]` to normalized action chunks

Main loss weights:

| Loss | Weight |
|---|---:|
| flow matching | 1.0 |
| inverse dynamics MSE | 1.0 |
| LeWM one-step consistency | 0.1 |
| latent smoothness | 0.0 |

All main models use:

```text
horizon = 5
action_block = 5
epochs = 10
batch_size = 128
```

TwoRoom:

```bash
python train_latent_planner.py \
  lewm_checkpoint=tworoom/lewm \
  data.dataset.name=tworoom \
  data.dataset.keys_to_load='[pixels,action,proprio]' \
  data.dataset.keys_to_cache='[action,proprio]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/tworoom_h5_ep10
```

PushT:

```bash
python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/pusht_h5_ep10
```

Reacher:

```bash
python train_latent_planner.py \
  lewm_checkpoint=reacher/lewm \
  data.dataset.name=reacher \
  data.dataset.keys_to_load='[pixels,action,observation]' \
  data.dataset.keys_to_cache='[action,observation]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/reacher_h5_ep10
```

OGBench Cube:

```bash
python train_latent_planner.py \
  lewm_checkpoint=cube/lewm \
  data.dataset.name=ogbench/cube_single_expert \
  data.dataset.keys_to_load='[pixels,action,observation]' \
  data.dataset.keys_to_cache='[action,observation]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/cube_h5_ep10
```

Training saves:

```text
$STABLEWM_HOME/<subdir>/latent_planner.pt
```

## SLURM Helpers

Example one-GPU training:

```bash
sbatch --export=ALL,TASK=pusht,HORIZON=5,EPOCHS=10,RUN_NAME=pusht_h5_ep10 \
  train_latent_planner.slurm
```

Example one-GPU evaluation:

```bash
sbatch --export=ALL,CONFIG_NAME=pusht.yaml,SOLVER=latent_flow,\
PLANNER_CHECKPOINT=leflow/pusht/latent_planner.pt,HORIZON=5,NUM_EVAL=50 \
  eval_latent_planner.slurm
```

## Key Files

| File | Purpose |
|---|---|
| [`latent_planner.py`](latent_planner.py) | LeFlow modules and solver wrapper |
| [`train_latent_planner.py`](train_latent_planner.py) | Training entry point |
| [`config/train/latent_planner.yaml`](config/train/latent_planner.yaml) | Training config |
| [`config/eval/solver/latent_flow.yaml`](config/eval/solver/latent_flow.yaml) | Evaluation solver config |
| [`eval.py`](eval.py) | LeWM-compatible evaluation entry point |

## Citation

This code builds on LeWorldModel:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}

@misc{huang2026leflowgenerativelatentflow,
      title={LeFlow: Generative Latent Flow Planning for World Models}, 
      author={Hsiang-Wei Huang and Jianxu Shangguan and Junbin Lu and Jenq-Neng Hwang},
      year={2026},
      eprint={2608.24855},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24855}, 
}
```
