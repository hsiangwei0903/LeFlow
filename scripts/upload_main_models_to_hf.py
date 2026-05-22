#!/usr/bin/env python3
"""Upload main H=5 latent planner checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


STABLEWM_HOME = Path("/gpfs/scrubbed/hwhuang/stable-wm")

CHECKPOINTS = {
    "tworoom": STABLEWM_HOME / "latent_planner/tworoom_h5_ep10/latent_planner.pt",
    "pusht": STABLEWM_HOME / "latent_planner/pusht_h5_ep10_24h/latent_planner.pt",
    "reacher": STABLEWM_HOME / "latent_planner/reacher_h5_ep10_24h/latent_planner.pt",
    "cube": STABLEWM_HOME / "latent_planner/cube_h5_ep10_24h/latent_planner.pt",
}


README = """---
license: mit
tags:
- world-model
- planning
- flow-matching
- robotics
- leworldmodel
---

# Latent Path Flow Planner Checkpoints

Main H=5 latent-path flow planner checkpoints trained on top of frozen LeWorldModel
backbones.

These checkpoints contain:

- latent planner state dict
- inverse dynamics state dict
- planner architecture metadata
- reference path/name for the frozen LeWM checkpoint

They do not serialize the full frozen LeWM object.

## Files

| Benchmark | Checkpoint |
|---|---|
| TwoRoom | `tworoom/latent_planner.pt` |
| PushT | `pusht/latent_planner.pt` |
| Reacher | `reacher/latent_planner.pt` |
| OGBench Cube | `cube/latent_planner.pt` |

## Expected Code

Use with the corresponding code release from the `le-wm` repository containing
`latent_planner.py`, `train_latent_planner.py`, and
`config/eval/solver/latent_flow.yaml`.

Example:

```bash
python eval.py --config-name=pusht.yaml \\
  solver=latent_flow \\
  policy=<downloaded-or-hub-checkpoint> \\
  plan_config.horizon=5
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. user/repo")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision to upload to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in CHECKPOINTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    api = HfApi()
    whoami = api.whoami()
    print(f"Uploading as {whoami.get('name', whoami)} to {args.repo_id}")

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for benchmark, path in CHECKPOINTS.items():
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            path_or_fileobj=str(path),
            path_in_repo=f"{benchmark}/latent_planner.pt",
        )
        print(f"uploaded {benchmark}: {path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        readme = Path(tmpdir) / "README.md"
        readme.write_text(README)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
        )
    print("upload complete")


if __name__ == "__main__":
    main()
