# Auto-labeling Tool

Utilities in this repository prepare RGB-only sequences for auto-labeling and YOLO dataset creation.

## Release Status

Current version: `v1.0.0`.

This version is centered on the `auto-label-tool` CLI and a config-driven RGB-to-YOLO workflow.

It supports mask creation, mask aggregation, dataset generation, inspection, and annotation curation. It does not promise portability, full automation, training-quality guarantees, stable APIs, or safe destructive dataset editing.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Docker Setup

The current repository uses the helper scripts in `docker/`; there is no Docker Compose file in this repo.

Build the image from the `docker/` directory:

```bash
cd docker
./docker_build.sh
```

Start or attach to the development container:

```bash
./docker_run.sh
```

The run script expects a Linux host with an NVIDIA GPU and X11 display forwarding. Edit `docker/docker_run.sh` if your model or dataset paths are different.

## Package Layout

```text
auto_labeling_tool/
  app/    # command-line workflows
  sam2/   # SAM2 propagation backends
  utils/  # shared image, tracking, and segmentation helpers
```

For the simplified CLI, install the repository in editable mode once inside the
container:

```bash
pip install -e .
```

Check the command is available:

```bash
auto-label-tool --help
```
## Pipeline
0. Create a configuration file
1. Create mask versions with manual prompts
2. Aggregate multiple mask versions into one `average_mask`.
3. Create a YOLO dataset from aggregated masks.
4. Validate or clean annotations interactively.
5. Visualize labels, report dataset stats, and reshuffle dataset splits.

```bash
auto-label-tool create-config -d /workspace/assets/samples -o /workspace/assets/runs
auto-label-tool create-masks -c /workspace/assets/runs/assets__samples/cfg.yaml
auto-label-tool aggregate-masks -c /workspace/assets/runs/assets__samples/cfg.yaml
auto-label-tool create-dataset -c /workspace/assets/runs/assets__samples/cfg.yaml
auto-label-tool validate -c /workspace/assets/runs/assets__samples/cfg.yaml
```

By default, `create-config` derives a data id from the last two input path
parts. For example, `assets/samples` becomes `assets__samples`, so `run.root`
and the default config path become `OUTPUT_DIR/assets__samples`.

The default full run executes mask creation, aggregation, and dataset creation:

```bash
auto-label-tool run -c /workspace/assets/runs/assets__samples/cfg.yaml
```

When a config file is moved, commands check whether `run.root` still matches the
directory containing the config. Interactive shells ask whether to update it or
cancel; non-interactive runs fail unless `--update-root` is passed. A pipeline
step never proceeds with a mismatched config. To fix the config without running
a pipeline step:

```bash
auto-label-tool update-root -c /workspace/assets/runs/assets__samples/cfg.yaml
```

Pipeline commands also accept `--update-root` to apply the same fix before
running.

To enable shell tab completion for the current Bash session:

```bash
eval "$(register-python-argcomplete auto-label-tool)"
```

Or install it automatically for future Bash sessions:

```bash
auto-label-tool install-completion
```

This appends the completion hook to `~/.bashrc` inside the environment where
`auto-label-tool` is installed.



## Expected Mask Layout

```text
<output-dir>/
  masks/
    <timestamp>/
      <frame_stem>.npy
      ...
```

Each `.npy` is a mask stack with shape `[n, H, W]`, with one slice per object.


Utility modes:

## 5) Visualize Labels

Use the dataset directory created under `paths.datasets_root`, for example:

```text
/workspace/assets/runs/assets__samples/datasets/20260514_093720
```

```bash
python3 -m auto_labeling_tool.app.visualize_dataset \
  --dataset-root /workspace/assets/runs/assets__samples/datasets/20260514_093720 \
  --split train \
  --fraction 0.1
```

Optional video export:

```bash
python3 -m auto_labeling_tool.app.visualize_dataset \
  --dataset-root /workspace/assets/runs/assets__samples/datasets/20260514_093720 \
  --split train \
  --output-video /tmp/train_vis.mp4 \
  --no-window
```

## 6) Dataset Stats

```bash
python3 -m auto_labeling_tool.app.dataset_stats \
  --dataset-dir /workspace/assets/runs/assets__samples/datasets/20260514_093720
```

Reports per-split counts for images, labels, missing/orphan labels, empty label files, total boxes, and boxes per class.

## 7) Shuffle Dataset Splits

Use `shuffle_dataset.py` to reshuffle an existing YOLO dataset into new
`train`, `val`, and optional `test` splits.

Preview the split counts without changing files:

```bash
python3 -m auto_labeling_tool.app.shuffle_dataset \
  --dataset-dir /workspace/assets/runs/assets__samples/datasets/20260514_093720 \
  --ratios 0.7 0.2 0.1 \
  --dry-run
```

Apply the reshuffle:

```bash
python3 -m auto_labeling_tool.app.shuffle_dataset \
  --dataset-dir /workspace/assets/runs/assets__samples/datasets/20260514_093720 \
  --ratios 0.7 0.2 0.1 \
  --seed 42
```

The command creates temporary backups while moving files and removes those
backups after a successful reshuffle.

