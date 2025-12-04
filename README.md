# RGB Autolabeling

Utilities in this workspace focus on preparing RGB-only datasets (no depth or lidar) for downstream autolabeling. 

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

## Batch Mask Creation (`create_masks.py`)

To generate and persist masks for an entire image sequence, use the `create_masks.py` script. It wraps the propagation demos into a non-interactive batch tool that saves compressed, lossless masks to disk while still letting you visualize overlays.

Example usage:

```bash
cd /workspace
python3 create_masks.py \
  /data/with_bed/rosbag2_2025_11_28-11_43_17 \
  --method sam2_prop
```

Key behavior:

- Input: a directory with an ordered list of RGB frames (e.g. rosbag extractions).
- Methods:
  - `sam2_prop` – standard SAM 2 video predictor.
  - `sam2_prop_recur` – recursive variant that resamples prompts from the propagated mask.
  - `sam2_lkt` – LK tracking + SAM 2 image predictor.
- Output masks are written as maximally compressed PNGs to:
  - `/data/masks/<sequence_dir.name>/<timestamp>/<original_frame_filename>`
  - e.g. for input frame  
    `/data/with_bed/rosbag2_2025_11_28-11_43_17/rosbag2_2025_11_28-11_43_17_1764301397533622123.jpg`  
    the mask path becomes  
    `/data/masks/rosbag2_2025_11_28-11_43_17/<timestamp>/rosbag2_2025_11_28-11_43_17_1764301397533622123.jpg`.
- During execution, an OpenCV window shows an overlay of the current mask on the RGB frame; press `q` or `ESC` to stop early.

## SAM 2 Video Propagation Demos

To interactively propagate a SAM 2 mask across a directory of RGB frames, use the top-level propagation scripts.

Basic usage (standard video predictor):

```bash
cd /workspace
python3 propagate_sam2.py \
  --sequence-dir /workspace/assets/seq_images \
  --checkpoint /models/sam2/sam2.1_hiera_large.pt \
  --model-config /models/sam2/configs/sam2.1/sam2.1_l.yaml
```

Recursive variant (periodically resamples prompt points from the propagated mask):

```bash
cd /workspace
python3 propagate_sam2_recursive.py \
  --sequence-dir /workspace/assets/seq_images \
  --checkpoint /models/sam2/sam2.1_hiera_large.pt \
  --model-config /models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

Key options (shared):

- `--sequence-dir` – directory with an ordered list of RGB frames (defaults to `/workspace/assets/seq_images`).
- `--checkpoint` – SAM 2 video checkpoint `.pt` (defaults to the large model under `/models/sam2`).
- `--model-config` – Hydra config or YAML path for the video predictor.
- `--device` – torch device for inference; defaults to `cuda:0`.
- `--async-loading` – stream frames asynchronously during initialization; **enabled by default**.
- `--offload-video-to-cpu` – keep frames on CPU in the inference state to reduce GPU memory; **enabled by default**.
- `--offload-state-to-cpu` – offload cached internal state to CPU to further reduce GPU footprint (slower).

For a lighter-weight alternative that combines SAM 2 segmentation with Lucas–Kanade keypoint tracking, see `propagate_sam2_lkt.py`, which uses the same sequence directory convention as the LK tracker example and runs the SAM 2 image predictor on `cuda:0` by default.
