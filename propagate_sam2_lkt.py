from pathlib import Path

import cv2
import numpy as np

from utils.lk_tracker import loop_iter_image_sequence, track_between_frames
from utils import get_segmentation, load_sam_model, overlay_mask
from utils.sampling_mask import jitter_mask_sampling as mask_sampling

SAM2_DEFAULT_DIR = Path("/models/sam2")
DEFAULT_SAM2_CONFIG = SAM2_DEFAULT_DIR / "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DEFAULT_DIR / "sam2.1_hiera_large.pt"


def get_initial_xy_points(image):
    window_name = "Select initial points"
    img_display = image.copy()
    selected = []

    def on_mouse(event, x, y, flags, param):  # type: ignore[unused-argument]
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((float(x), float(y)))
            cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(window_name, img_display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, img_display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27, 32):  # ENTER, ESC, SPACE
            break

    cv2.destroyWindow(window_name)
    return selected


def get_sam_model(checkpoint, model_config=DEFAULT_SAM2_CONFIG):
    return load_sam_model(
        checkpoint_path=str(checkpoint),
        model_config=str(model_config),
        device="cuda:0",
    )


def compute_seg_mask(points_xy, frame, sam_model):
    mask, _ = get_segmentation(
        image=frame,
        keypoints_uv=points_xy,
        predictor=sam_model,
        multimask_output=False,
    )
    overlay = overlay_mask(frame, mask)
    overlay_disp = overlay.copy()
    if points_xy:
        for u, v in points_xy:
            cv2.circle(
                overlay_disp,
                (int(round(u)), int(round(v))),
                3,
                (0, 255, 0),
                -1,
            )
    return mask, overlay_disp



def track_xy(points_xy, frame, prev_frame):
    if not points_xy:
        return []
    next_points, statuses = track_between_frames(prev_frame, frame, points_xy)
    statuses = np.asarray(statuses, dtype=bool)
    return [
        (float(u), float(v))
        for (u, v), ok in zip(next_points, statuses)
        if ok
    ]

def main():
    sequence_dir = Path("/workspace/assets/seq_images")
    _xy = None
    checkpoint = DEFAULT_SAM2_CHECKPOINT

    sam = get_sam_model(checkpoint)
    _fr = None
    for fr in loop_iter_image_sequence(sequence_dir):
        if _xy is None:
            _xy = get_initial_xy_points(fr)
            mask, ov_maks = compute_seg_mask(_xy, fr, sam)
            cv2.imshow("overlay", ov_maks)
            _xy = mask_sampling(mask, _xy, 10)
            _fr = fr
            continue
        
        _xy = track_xy(_xy, fr, _fr)
        if len(_xy) == 0:
            _xy = None
            continue
        _fr = fr
        mask, ov_maks = compute_seg_mask(_xy, fr, sam)
        _xy = mask_sampling(mask, _xy, 10)
        cv2.imshow("overlay", ov_maks)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
