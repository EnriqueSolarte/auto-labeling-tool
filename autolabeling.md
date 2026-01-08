# Autolabeling Pipeline

## Overview
Auto-label a sequence of RGB images using SAM2-based propagation. You manually select an initial frame, click prompts, and the mask is propagated across the sequence. Multiple runs can be aggregated for a more stable result.

## Prerequisites
- A directory of RGB frames (jpg/png/etc). Filenames can be numeric or not.
- SAM2 weights and config available at their default paths, or ensure your environment is already configured for the scripts.
- GUI access (OpenCV windows are used for clicks and previews).

## Directory layout
Example input sequence:
```
seq_images/
  1764302222276118032.jpg
  1764302222305435226.jpg
  1764302222338964634.jpg
  1764302222371981102.jpg
```

Output structure created by `create_masks_by_manual_prompts.py`:
```
<output-dir>/
  <sequence_name>/
    <timestamp>/
      <frame_stem>.npy
      ...
```
Each `.npy` is a stack of binary masks with shape `[n, H, W]` where `n` is the number of objects in that frame.

## Step 1: Create masks with manual prompts
Run the script and choose a backend:
```sh
python3 /workspace/create_masks_by_manual_prompts.py \
  --sequence-dir /workspace/assets/seq_images \
  --method sam2_prop \
  --output-dir /workspace/assets/masks
```

### What you will do in the UI
- Frame selection window:
  - `SPACE` to advance to the next frame
  - `ENTER` to select the frame
  - `ESC`/`q` to cancel
- Prompt selection window:
  - Left-click to add prompt points
  - `ENTER`, `SPACE`, or `ESC` to finish selection

Notes:
- Run the script multiple times with different initial frames and prompts. Each run writes a new timestamped version under the same sequence directory.
- The script supports multiple objects: if you create multiple masks, they are saved in one `.npy` per frame.

## Step 2: Aggregate multiple mask versions
Aggregate all versions for consistency:
```sh
python3 /workspace/aggregate_masks.py \
  --masks-dir /workspace/assets/masks/seq_images \
  --debug /workspace/assets/seq_images
```

Outputs:
```
<masks-dir>/<sequence_name>/average_mask/
  <frame_stem>.npy   # stacked per-object masks
  <frame_stem>.png   # union mask for quick visualization
```

Options:
- `--thr 0.5` controls the mean threshold for binarization.
- Omit `--debug` in headless environments.

## Step 3: Build a YOLO training dataset
Create a training dataset (images + labels + data.yaml) from the aggregated masks:
```sh
python3 /workspace/create_training_dataset_from_masks.py \
  --masks_dir /workspace/assets/masks \
  --images_dir /workspace/assets/seq_images \
  --output-root /workspace/assets/training_data \
  --class-name bed \
  --val-ratio 0.3 \
  --seed 42
```

Important details:
- This script only accepts masks under an `average_mask/` directory. If none exist under `--masks_dir`, it will refuse to process other masks.
- The script matches masks to images by filename stem; extensions can differ.
- Images are symlinked into the dataset when possible, otherwise copied.

Outputs:
```
<output-root>/<timestamp>/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```
`labels/*.txt` are YOLO boxes derived from each mask stack.

## Step 4: Visualize the training dataset
Quickly inspect labels overlaid on images:
```sh
python3 /workspace/visualize_training_dataset.py \
  --dataset-root /workspace/assets/training_data/<timestamp> \
  --split train \
  --fraction 0.1
```

Controls:
- `SPACE`: next image
- `Q` or `ESC`: quit
