# Autolabeling Pipeline

## Overview
This workflow builds a YOLO dataset from RGB sequences using SAM2 masks, mask aggregation, and annotation validation.

## Prerequisites
- RGB frame directory (jpg/png/etc.)
- SAM2 checkpoint + config available in your environment
- OpenCV GUI support for interactive steps

## Expected Mask Layout
`create_masks_by_manual_prompts.py` writes:

```text
<output-dir>/
  <sequence_name>/
    <timestamp>/
      <frame_stem>.npy
      ...
```

Each `.npy` is a mask stack with shape `[n, H, W]` (one slice per object).

## Step 1: Create Masks (Interactive)

```bash
python3 /workspace/create_masks_by_manual_prompts.py \
  --sequence-dir /workspace/assets/seq_images \
  --method sam2_prop \
  --output-dir /workspace/assets/masks
```

UI controls:
- Frame selection: `SPACE` next, `ENTER` select, `ESC/q` cancel
- Prompt selection: left click points, `ENTER/SPACE/ESC` finish

Notes:
- Repeat this step multiple times to generate multiple timestamped mask versions.
- Supported methods: `sam2_prop`, `sam2_prop_recur`, `sam2_lkt`.

## Step 2: Aggregate Mask Versions

```bash
python3 /workspace/aggregate_masks.py \
  --masks-dir /workspace/assets/masks \
  --thr 0.5 \
  --debug /workspace/assets/seq_images
```

Outputs:

```text
<masks-dir>/<sequence_name>/average_mask/
  <frame_stem>.npy
  <frame_stem>.png
```

Options:
- `--thr`: threshold on mean mask
- `--init-idx`: resume from sorted frame index (keeps existing outputs)
- `--debug [RGB_DIR]`: preview aggregated masks with optional RGB overlay

## Step 3: Build YOLO Dataset

```bash
python3 /workspace/create_training_dataset_from_masks.py \
  --masks_dir /workspace/assets/masks \
  --images_dir /workspace/assets/seq_images \
  --output-root /workspace/assets/training_data \
  --class-name bed \
  --val-ratio 0.3 \
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

## Step 4: Validate and Clean Annotations

Interactive review:

```bash
python3 /workspace/validate_dataset_annotations.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --int-idx 0
```

Viewer controls:
- `x`: blacklist current image (`black_list.txt`)
- `a`: remove from blacklist
- `j`: previous
- `l`: next
- `q` or `ESC`: quit

Utility modes:

```bash
# create missing empty label files
python3 /workspace/validate_dataset_annotations.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --ensure-labels

# delete blacklisted images and matching labels
python3 /workspace/validate_dataset_annotations.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --prune
```

## Step 5: Visualize Labels

```bash
python3 /workspace/visualize_dataset.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --fraction 0.1
```

Optional video export:

```bash
python3 /workspace/visualize_dataset.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --output-video /tmp/train_vis.mp4 \
  --no-window
```

## Step 6: Dataset Stats

```bash
python3 /workspace/dataset_stats.py \
  --dataset-dir /workspace/assets/training_data/<timestamp>
```

Reports per-split counts for images, labels, missing/orphan labels, empty label files, and boxes per class.
