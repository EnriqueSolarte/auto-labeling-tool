Date: 2026-05-14
Project version: v0.1.0
Description: Simple current workflow for creating masks and YOLO datasets.

# How to Use the Current Pipeline

This is the practical workflow for the current implementation.

## 1. Create masks with manual prompts

```bash
python3 create_masks_by_manual_prompts.py \
  --sequence-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/ \
  --output-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/
```

An image window will open.

1. Select the initial frame.
2. Add prompt points inside the object.
3. The system propagates the mask through the sequence.
4. Masks are saved under:

```text
/data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/<timestamp>
```

Example output:

```text
masks/
`-- cam_0
    |-- 20260514_054016
    `-- 20260514_054309
```

Run this step multiple times if you want several mask versions for the same sequence.

## 2. Aggregate mask versions

The goal is to combine mask versions from multiple manual runs into one averaged mask set.

```bash
python3 aggregate_masks.py \
  --masks-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/ \
  --debug /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/
```

All mask versions under `masks/cam_0/` are aggregated into `average_mask`.

```text
masks/
`-- cam_0
    |-- 20260514_054016
    |-- 20260514_054309
    `-- average_mask
```

## 3. Create an Ultralytics YOLO dataset

```bash
python3 create_training_dataset_from_masks.py \
  --masks_dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/average_mask/ \
  --images_dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/ \
  --output-root /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/ \
  --class-name bamboo
```

Example output:

```text
labels/
`-- 20260514_055605
    |-- data.yaml
    |-- images
    `-- labels
```

## 4. Check dataset stats

```bash
python3 dataset_stats.py \
  --dataset-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/20260514_055605/
```

## 5. Reshuffle a dataset

```bash
python3 manipulate_dataset.py \
  --dataset-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/20260514_055605/ \
  --ratios 0.5 0.25
```
