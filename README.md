# RGB Autolabeling

Utilities in this workspace focus on preparing RGB-only datasets (no depth or lidar) for downstream autolabeling. 

## Collecting Images From Rosbags

Recorded missions should be stored as standard ROS 2 `rosbag2` folders (each containing `metadata.yaml` plus the `.mcap` file). The `extract_rosbag_images.py` helper reads `/camera/camera/color/image_raw` frames from every bag and writes them to `/data/<bag-name>/`. Encoding support currently covers `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, and `8uc1`.

Before running the script, be sure to source your ROS 2 environment (e.g. `source /opt/ros/jazzy/setup.bash`) so the `rosbag2_py` dependencies are available.

## Usage

The script provides two entry points—one for a single bag and one for a directory of bags. Examples:

```bash
# Export frames from one bag
python3 /workspace/extract_rosbag_images.py \
  --bag /rosbag/no_bed/rosbag2_2025_11_28-10_56_12 \
  --output-dir /data

# Export frames from every bag inside a directory
python3 /workspace/extract_rosbag_images.py \
  --bag-dir /rosbag/no_bed \
  --output-dir /data
```

Optional flags:

- `--topic` – override the image topic (default `/camera/camera/color/image_raw`).
- `--limit` – cap the number of frames per bag (e.g. `--limit 100`).
- `--log-level` – adjust verbosity (`DEBUG`, `INFO`, etc.).
- `--verbose` – stream per-frame extraction progress to the console.

## Interactive LK Keypoint Tracking

For quick visual inspection of motion across a sequence of RGB frames, the `utils.lk_tracker` module provides an interactive Lucas–Kanade (LK) tracker. It expects a directory of images (e.g. the extracted rosbag frames) and lets you click UV points on the first frame, then visualizes how those points move through the entire sequence.

Run the interactive demo via the LK example script:

```bash
cd /workspace
python3 -m examples.lk_tracker_example --sequence-dir /workspace/assets/seq_images
```

Workflow:

- The script loads and sorts all images in `--sequence-dir`.
- It displays the first frame and lets you add UV points with left clicks.
- After you press SPACE/ENTER/ESC, it tracks those keypoints through all frames with LK optical flow.
- A viewer window then loops through the sequence forward and backward (“ping-pong”), overlaying the tracked points as red dots. Press `ESC` or `q` to exit.

Requirements:

- Python `opencv-python` (`cv2`) must be installed on your environment.
- A display is required for the interactive windows (e.g. running locally or with X forwarding).

## SAM 2 Quick Segmentation Example

For a minimal Segment Anything Model 2 (SAM 2) sanity check, use the standalone example script in `examples/`. By default it reads `/workspace/assets/image.jpg`, writes a binary mask `mask.png`, and an RGB overlay image next to the input.

Run the example:

```bash
cd /workspace
python3 -m examples.sam_example \
  --points "320,200 360,230 300,260" \
  --checkpoint /models/sam2/sam2.1_hiera_tiny.pt \
  --model-config /models/sam2/sam2.1_hiera_t.yaml
```

Key options:

- `--image` – path to the RGB frame to segment (defaults to `/workspace/assets/image.jpg`).
- `--checkpoint` – path to the SAM 2 `.pt` checkpoint (defaults to `/models/sam2/sam2.1_hiera_tiny.pt`).
- `--model-config` – Hydra config for the checkpoint (defaults to `/models/sam2/sam2.1_hiera_t.yaml` when present, otherwise the packaged tiny config).
- `--points` – space-delimited UV points `"x,y"` that seed the segmentation.
- `--output-mask` – destination for the mask PNG (defaults to `<image-dir>/mask.png`).
- `--output-overlay` – destination for the overlay image (defaults to `overlay_<image-name>`).
