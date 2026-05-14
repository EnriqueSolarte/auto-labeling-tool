# Auto-labeling Tool

Utilities in this repository prepare RGB-only sequences for auto-labeling and YOLO dataset creation.

## Release Status

Current target: `v1.0.0`.

This version is intended for internal teammates running the manual RGB-to-YOLO pipeline inside the provided Docker GPU environment, using SAM2 mounted from the host and datasets mounted under `/data`.

It supports mask creation, mask aggregation, dataset generation, inspection, and annotation curation. It does not promise portability, full automation, training-quality guarantees, stable APIs, or safe destructive dataset editing.

Before tagging or using this version, read `.agent-workspace/release-checklist.md` and `.agent-workspace/known-issues.md`.

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

The run script expects a Linux host with an NVIDIA GPU and X11 display forwarding. It starts an `auto-labeling:dev` container named `auto-labeling-dev`, mounts this repository at `/workspace`, and uses `/workspace` as the working directory.

The current hard-coded mounts in `docker/docker_run.sh` are:

- `/media/q200/models_weights` mounted at `/models`
- `/media/v1-data/panda_ai_dataset` mounted at `/data/panda`
- `/media/v1-data/hospital-bishop-data` mounted at `/data/bishop`

Edit `docker/docker_run.sh` if your model or dataset paths are different.

## Package Layout

```text
auto_labeling_tool/
  app/    # command-line workflows
  sam2/   # SAM2 propagation backends
  utils/  # shared image, tracking, and segmentation helpers
```

Run workflows from the repository root with `python3 -m ...`.

For the simplified CLI, install the repository in editable mode once inside the
container:

```bash
pip install -e .
```

Then create a run config and drive the workflow with one command family:

```bash
auto-label-tool create-config -d /data/panda/panda-v3-data/panda-v3-2026-05-12/_0/cam_0 -o /data/panda/panda-v3-data/panda-v3-2026-05-12/data
auto-label-tool create-masks -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
auto-label-tool aggregate-masks -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
auto-label-tool create-dataset -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
auto-label-tool validate -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
```

By default, `create-config` derives a data id from the last two input path
parts. For example, `_0/cam_0` becomes `_0__cam_0`, so `run.root` and the
default config path become `OUTPUT_DIR/_0__cam_0`.

The default full run executes mask creation, aggregation, and dataset creation:

```bash
auto-label-tool run -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
```

When a config file is moved, commands check whether `run.root` still matches the
directory containing the config. Interactive shells ask whether to update it or
cancel; non-interactive runs fail unless `--update-root` is passed. A pipeline
step never proceeds with a mismatched config. To fix the config without running
a pipeline step:

```bash
auto-label-tool update-root -c /data/panda/panda-v3-data/panda-v3-2026-05-12/data/_0__cam_0/cfg.yaml
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

## Pipeline

1. Create mask versions with manual prompts
2. Aggregate multiple mask versions into one `average_mask`.
3. Create a YOLO dataset from aggregated masks.
4. Validate or clean annotations interactively.
5. Visualize labels and report dataset stats.

## Prerequisites

- RGB frame directory (`.jpg`, `.png`, etc.)
- SAM2 checkpoint and config available in your environment
- OpenCV GUI support for interactive steps

## Expected Mask Layout

`auto_labeling_tool.app.create_masks_by_manual_prompts` writes:

```text
<output-dir>/
  <sequence_name>/
    <timestamp>/
      <frame_stem>.npy
      ...
```

Each `.npy` is a mask stack with shape `[n, H, W]`, with one slice per object.

## 1) Create Masks

```bash
python3 -m auto_labeling_tool.app.create_masks_by_manual_prompts \
  --sequence-dir /data/with_bed/rosbag2_2025_11_28-11_43_17 \
  --method sam2_prop \
  --output-dir /data/masks
```

UI controls:

- Frame selection: `SPACE` next, `ENTER` select, `ESC/q` cancel
- Prompt selection: left click points, `ENTER/SPACE/ESC` finish

Notes:

- Repeat this step multiple times to generate multiple timestamped mask versions.
- Supported methods: `sam2_prop`, `sam2_prop_recur`, `sam2_lkt`.

## 2) Aggregate Mask Versions

```bash
python3 -m auto_labeling_tool.app.aggregate_masks \
  --masks-dir /data/masks \
  --thr 0.5 \
  --debug /data/with_bed
```

Outputs:

```text
<masks-dir>/<sequence_name>/average_mask/
  <frame_stem>.npy
  <frame_stem>.png
```

Options:

- `--thr`: threshold on mean mask
- `--init-idx`: resume from sorted frame index, keeping existing outputs
- `--debug [RGB_DIR]`: preview aggregated masks with optional RGB overlay

Resume long runs from an index:

```bash
python3 -m auto_labeling_tool.app.aggregate_masks --masks-dir /data/masks --init-idx 500
```

## 3) Build YOLO Dataset

```bash
python3 -m auto_labeling_tool.app.create_training_dataset_from_masks \
  --masks_dir /data/masks \
  --images_dir /data/with_bed \
  --output-root /data/training_data \
  --class-name bed \
  --val-ratio 0.2 \
  --seed 42
```

Important behavior:

- Only `.npy` masks are used.
- If any `average_mask` directories exist, only those are used.
- If none exist under `--masks_dir`, the script stops instead of processing raw versions.

Outputs:

```text
<output-root>/<timestamp>/
  data.yaml
  images/train
  images/val
  labels/train
  labels/val
```

## 4) Validate and Clean Annotations

Interactive review:

```bash
python3 -m auto_labeling_tool.app.validate_dataset_annotations \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --int-idx 0
```

Viewer controls:

- `x`: blacklist current image in `black_list.txt`
- `a`: remove current image from blacklist
- `j`: previous frame
- `l`: next frame
- `q` or `ESC`: quit

Utility modes:

```bash
# create empty label files for images missing .txt
python3 -m auto_labeling_tool.app.validate_dataset_annotations \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --ensure-labels

# delete blacklisted images and matching labels
python3 -m auto_labeling_tool.app.validate_dataset_annotations \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --prune
```

## 5) Visualize Labels

```bash
python3 -m auto_labeling_tool.app.visualize_dataset \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --fraction 0.1
```

Optional video export:

```bash
python3 -m auto_labeling_tool.app.visualize_dataset \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --output-video /tmp/train_vis.mp4 \
  --no-window
```

## 6) Dataset Stats

```bash
python3 -m auto_labeling_tool.app.dataset_stats \
  --dataset-dir /data/training_data/20260113_065116
```

Reports per-split counts for images, labels, missing/orphan labels, empty label files, total boxes, and boxes per class.

## SAM2 Propagation Demos

These lower-level demos are mostly useful for backend development and comparison:

```bash
python3 -m auto_labeling_tool.sam2.propagate_sam2 --sequence-dir /workspace/assets/seq_images
python3 -m auto_labeling_tool.sam2.propagate_sam2_recursive --sequence-dir /workspace/assets/seq_images
python3 -m auto_labeling_tool.sam2.propagate_sam2_lkt --sequence-dir /workspace/assets/seq_images
```
