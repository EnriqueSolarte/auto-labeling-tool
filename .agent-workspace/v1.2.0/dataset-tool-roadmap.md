Date: 2026-05-18
Project version: v1.1.0
Description: Dataset creation tool improvements are split into focused PRs.

# Discussion: `create_training_dataset_from_masks.py`

## Current Assessment

The script currently creates a basic single-class YOLO detection dataset from
mask files and RGB images.

Brutally honest summary:

- It uses masks as input, but outputs bounding-box labels only.
- It does not create segmentation or mask labels.
- It does not preserve masks in the generated dataset.
- It uses all matched image/mask pairs.
- It only randomizes the train/val split.
- It does not support test, valid, split-1, split-2, or arbitrary split names.
- It does not support random subsampling of the raw matched data.
- It advertises `.png` mask support in the help text, but actual mask discovery
  only includes `.npy`.
- It assumes a single class and always writes class id `0`.
- It may silently skip bad or empty masks while still linking/copying images.

## Recommended Engineering Workflow

The correct professional workflow is:

1. Create an issue.
2. Create a branch for that issue.
3. Implement the fix with tests and documentation updates.
4. Open a pull request.
5. Review the pull request.
6. Merge after approval.
7. Create a new version or changelog entry if this tool is packaged or used by
   other users.

Important note: this should not be treated as one huge issue. The requested
changes mix bug fixes, behavior changes, and new features. They should be split
into multiple focused issues and pull requests.

## Recommended Issue Breakdown

### 1. Clarify Supported Mask Input Format

Decision:

- Only `.npy` masks are supported as true input masks.
- `.png` masks are visualization artifacts only.

Required changes:

- Update CLI help text.
- Update docs/comments.
- Make the behavior explicit and not misleading.

This is a small bug/documentation issue and should be fixed first.

### 2. Add Dataset Subsampling

Goal:

- Allow the user to create a dataset from only a percentage or count of the raw
  matched image/mask pairs.

Possible options:

```bash
--sample-ratio 0.25
--sample-count 500
--seed 42
```

Expected behavior:

- Collect all matched samples.
- Shuffle deterministically using the seed.
- Select the requested subset.
- Split the subset into the requested dataset splits.

Tests should verify:

- The same seed gives the same selected samples.
- Different seeds can produce different selections.
- Invalid ratios/counts fail clearly.
- Sampling happens before splitting.

### 3. Generalize Dataset Splits

Goal:

- Support arbitrary split names and ratios.

Possible options:

```bash
--splits train val test
--ratios 0.7 0.2 0.1
```

Default behavior:

```bash
--splits train val
--ratios 0.7 0.3
```

Required behavior:

- Validate that the number of split names matches the number of ratios.
- Validate that all ratios are non-negative.
- Validate that at least one ratio is greater than zero.
- Normalize ratios if they do not sum exactly to `1.0`.
- Write images and labels under each requested split.
- Generate `data.yaml` dynamically with all supported splits.

This should replace the current hard-coded train/val logic.

### 4. Add Copy vs Symlink Mode

Goal:

- Let users choose whether output images are symlinked or copied.

Possible option:

```bash
--image-mode symlink
--image-mode copy
```

Default:

```bash
--image-mode symlink
```

This is small and can be a focused pull request.

### 5. Add BBox vs Mask Label Output

Goal:

- Allow users to choose the label format.

Possible option:

```bash
--label-format bbox
--label-format mask
```

Important design question:

What does `mask` mean?

Possible interpretations:

- YOLO segmentation polygon labels.
- Copied `.npy` mask files.
- PNG mask files.
- COCO-style segmentation annotations.

This must be decided before implementation. This is the largest feature and
should not be mixed into the split/sampling pull requests.

Recommended first implementation:

- Keep `bbox` as the default.
- If supporting YOLO segmentation, convert each binary mask slice into polygon
  coordinates and write YOLO segmentation `.txt` labels.
- Add tests using simple synthetic masks.

## Recommended Pull Request Order

### PR 1: Cleanup and Documentation Truth

Scope:

- Fix misleading `.png` mask input help text.
- Clarify that `.npy` is the supported input format.
- Add or update tests around mask file discovery.

Risk:

- Low.

### PR 2: Flexible Split Support

Scope:

- Replace `_split_train_val()` with generic split logic.
- Add `--splits` and `--ratios`.
- Generate `data.yaml` from the selected split names.
- Add tests.

Risk:

- Medium.

### PR 3: Raw Dataset Subsampling

Scope:

- Add `--sample-ratio` and/or `--sample-count`.
- Apply sampling before splitting.
- Make sampling deterministic through `--seed`.
- Add tests.

Risk:

- Medium.

### PR 4: Copy/Symlink Mode

Scope:

- Add `--image-mode symlink|copy`.
- Preserve current behavior by defaulting to `symlink`.
- Add tests for both modes where practical.

Risk:

- Low.

### PR 5: Segmentation/Mask Label Support

Scope:

- Add `--label-format bbox|mask`.
- Define the exact mask output format.
- Implement conversion/writing.
- Add tests and documentation.

Risk:

- High.

This PR should happen after the output format is explicitly decided.

## Versioning Guidance

A new version is necessary if this tool is packaged, distributed, or used by
other people through a release process.

If this is an internal script only, a changelog entry or clear PR description
may be enough.

If semantic versioning is used:

- Documentation-only correction: patch version.
- Backward-compatible CLI additions: minor version.
- Breaking behavior changes: major version.

## Final Recommendation

Do not implement all requested changes in one pull request.

Start with the misleading `.png` mask documentation issue, then add flexible
splits, then add deterministic subsampling, then add copy/symlink mode, and only
after that design and implement mask/segmentation label output.
