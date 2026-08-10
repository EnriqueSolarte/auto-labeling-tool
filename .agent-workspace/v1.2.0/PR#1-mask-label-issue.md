Date: 2026-05-18
Project version: v1.2.0
Description: PR#1 issue must lock YOLO segmentation output before implementation.

# Issue: Add YOLO Segmentation Label Output To Dataset Creation

## Title

`feat: add YOLO segmentation label output for mask-derived datasets`

## Brutal Context

The dataset creation tool currently accepts `.npy` mask stacks but writes only
YOLO detection bounding boxes. That is misleading for a mask-based workflow:
the source annotation is segmentation, but the generated dataset throws away
shape information.

This issue must not become a vague "support masks" task. The word `mask` is
ambiguous and dangerous here. For this PR, `mask` means exactly one thing:
YOLO segmentation polygon labels written in `.txt` files.

## Problem

`auto_labeling_tool/app/create_training_dataset_from_masks.py` currently:

- discovers only `.npy` masks;
- loads each mask stack as `[n, H, W]`;
- converts each non-empty mask slice into one YOLO bbox line;
- writes labels as `class_id cx cy width height`;
- provides no CLI/config option to preserve segmentation geometry.

This blocks training YOLO segmentation models from the masks already produced
by the pipeline.

## Scope

Add a label-format option:

```bash
--label-format bbox
--label-format mask
```

Required behavior:

- `bbox` remains the default and preserves existing output.
- `mask` writes YOLO segmentation labels:
  `class_id x1 y1 x2 y2 ... xn yn`
- Coordinates are normalized to `[0, 1]`.
- One non-empty mask slice produces one label row.
- Empty or invalid masks are skipped with explicit, reviewable behavior.
- `.npy` remains the only supported input mask format.

## Out Of Scope

- COCO JSON export.
- Copying `.npy` masks into the dataset.
- PNG mask labels.
- Multi-class mapping.
- Split generalization.
- Raw dataset subsampling.
- Copy/symlink mode.
- Rewriting validation, visualization, or stats tools to fully understand
  segmentation labels.

## Acceptance Criteria

- Running without `--label-format` produces bbox labels exactly as before.
- Running with `--label-format bbox` produces bbox labels exactly as before.
- Running with `--label-format mask` produces YOLO segmentation polygon labels.
- The implementation handles stacked masks, where each slice is one object.
- Synthetic tests cover:
  - bbox default behavior;
  - simple rectangular mask to polygon label;
  - empty mask skip behavior;
  - invalid mask shape skip behavior;
  - deterministic output for the same mask.
- CLI help text states `.npy` masks are the supported input source.
- README or relevant docs mention the new label format option.
- Python syntax checks pass for touched scripts.

## Engineering Notes

OpenCV contour extraction is acceptable for polygon generation if the behavior
is deterministic and tested. The implementation should choose the largest
external contour for each mask slice unless a better documented policy is
implemented.

Do not invent abstractions for future COCO/multiclass export. This PR is only
about `bbox|mask` label writing.

## Review Risks

- Polygon simplification can degrade annotation quality if too aggressive.
- YOLO segmentation requires at least three points; tiny masks may not produce
  valid polygons.
- Current downstream tools parse the first five values as boxes. They may not
  correctly inspect segmentation labels yet. That limitation must be documented
  rather than hidden.

## Suggested Branch

`feat/yolo-segmentation-labels`

## Suggested PR Title

`feat: add YOLO segmentation label output`

## Suggested Checks

```bash
python3 -m py_compile auto_labeling_tool/app/create_training_dataset_from_masks.py
```

If tests are added:

```bash
python3 -m pytest
```
