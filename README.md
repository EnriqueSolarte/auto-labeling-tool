# RGB Autolabeling

Utilities in this repository prepare RGB-only sequences for autolabeling and YOLO dataset creation.

## Pipeline (Current)

1. Create mask versions with manual prompts + SAM2 propagation.
2. Aggregate multiple mask versions into one `average_mask`.
3. Create a YOLO dataset from aggregated masks.
4. Validate/clean annotations interactively.
5. Visualize and report dataset stats.

For a concise walkthrough, see `autolabeling.md`.

## 1) Create Masks

```bash
python3 create_masks_by_manual_prompts.py \
  --sequence-dir /data/with_bed/rosbag2_2025_11_28-11_43_17 \
  --method sam2_prop \
  --output-dir /data/masks
```

- Output: `.npy` mask stacks per frame under `/data/masks/<episode>/<timestamp>/`.
- Methods:
  - `sam2_prop`
  - `sam2_prop_recur`
  - `sam2_lkt`

## 2) Aggregate Mask Versions

```bash
python3 aggregate_masks.py \
  --masks-dir /data/masks \
  --thr 0.5 \
  --debug /data/with_bed
```

- Aggregates all versions per frame using mean + threshold.
- Writes outputs to `/data/masks/<episode>/average_mask/`:
  - `<frame>.npy` (stack `[n, H, W]`)
  - `<frame>.png` (union visualization)
- Resume long runs from an index:

```bash
python3 aggregate_masks.py --masks-dir /data/masks --init-idx 500
```

## 3) Build YOLO Dataset

```bash
python3 create_training_dataset_from_masks.py \
  --masks_dir /data/masks \
  --images_dir /data/with_bed \
  --output-root /data/training_data \
  --val-ratio 0.2 \
  --seed 42 \
  --class-name bed
```

- Uses `.npy` masks (and prefers `average_mask` paths when present).
- Creates timestamped dataset root:
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
  - `data.yaml`

## 4) Validate / Clean Annotations

Interactive validator:

```bash
python3 validate_dataset_annotations.py \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --int-idx 0
```

Key controls in viewer:

- `x`: add current image to `black_list.txt`
- `a`: remove current image from blacklist
- `j`: previous frame
- `l`: next frame
- `q`/`ESC`: quit

Utility modes:

```bash
# create empty label files for images missing .txt
python3 validate_dataset_annotations.py --dataset-root /data/training_data/20260113_065116 --split train --ensure-labels

# prune blacklisted images and matching labels
python3 validate_dataset_annotations.py --dataset-root /data/training_data/20260113_065116 --split train --prune
```

## 5) Visualize Dataset

```bash
python3 visualize_dataset.py \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --fraction 0.1
```

Optional video export:

```bash
python3 visualize_dataset.py \
  --dataset-root /data/training_data/20260113_065116 \
  --split train \
  --output-video /tmp/train_vis.mp4 \
  --no-window
```

## 6) Dataset Statistics

```bash
python3 dataset_stats.py --dataset-dir /data/training_data/20260113_065116
```

- Reports split-level counts:
  - images, labels, missing labels, orphan labels
  - empty label files
  - total boxes and boxes per class

## Additional Demos

### LK tracker demo

```bash
python3 -m examples.lk_tracker_example --sequence-dir /workspace/assets/seq_images
```

### SAM2 quick image segmentation demo

```bash
python3 -m examples.sam_example \
  --points "320,200 360,230 300,260" \
  --checkpoint /models/sam2/sam2.1_hiera_tiny.pt \
  --model-config /models/sam2/sam2.1_hiera_t.yaml
```

### SAM2 propagation demos

```bash
python3 propagate_sam2.py --sequence-dir /workspace/assets/seq_images
python3 propagate_sam2_recursive.py --sequence-dir /workspace/assets/seq_images
python3 propagate_sam2_lkt.py --sequence-dir /workspace/assets/seq_images
```
