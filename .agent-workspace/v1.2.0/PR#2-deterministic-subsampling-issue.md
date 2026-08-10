Date: 2026-05-18
Project version: v1.2.0
Description: issue must add deterministic raw dataset subsampling before train/val splitting.

# Issue: Add Deterministic Raw Dataset Subsampling

## Title

`feat: add deterministic dataset subsampling`

## Brutal Context

The dataset creation tool currently uses every matched `.npy` mask and RGB image
pair. The only randomness is the train/val split. That means users cannot build
a smaller reproducible training dataset from a large raw mask set without
manually copying files or relying on external scripts.

This issue must not become "dataset split redesign." Subsampling and split
generalization are different features. For this PR, sampling means exactly one
thing: select a deterministic percentage of the raw matched samples before the
existing train/val split runs.

## Problem

`auto_labeling_tool/app/create_training_dataset_from_masks.py` currently:

- discovers all matched mask/image pairs;
- uses every matched pair;
- shuffles only for train/val assignment;
- exposes no `--sample-ratio` option;
- has no config field for raw dataset sampling;

The documented workflow also uses `auto-label-tool create-dataset`, implemented
in `auto_labeling_tool/app/cli.py`. Updating only the standalone script would be
an incomplete fix.

## Scope

Add percentage-based raw sample selection:

```bash
--sample-ratio 0.25
```

Required behavior:

- Sampling happens after matching masks to images.
- Sampling happens before train/val splitting.
- `None` or an omitted sample ratio preserves the current behavior and uses all
  matched samples.
- Valid sample ratios are greater than `0` and less than or equal to `1`.
- The same sample ratio and seed select the same raw samples.
- Different seeds can select different raw samples.
- Sampling does not depend on nondeterministic filesystem traversal order.
- Sampling works identically for `bbox` and `mask` label formats.
- The documented `auto-label-tool create-dataset` workflow supports the same
  behavior as the standalone script.

Recommended implementation:

- Add a shared helper such as `_subsample_samples(samples, sample_ratio, seed)`.
- Sort samples by a stable path-based key before random selection.
- Use the existing `seed` setting as the deterministic control.
- Use `ceil(len(samples) * sample_ratio)` so a valid nonzero ratio does not
  silently select zero samples.
- Return selected samples in stable order after selection, then let the existing
  train/val split perform split shuffling.

## Out Of Scope

- `--sample-count`.
- Arbitrary split names.
- Split ratio generalization.
- Copy versus symlink mode.
- Dataset merge, shuffle, stats, or visualization changes.
- Multi-class support.
- Label format changes.
- Manifest writing, unless it is tiny and directly useful for review.

## Acceptance Criteria

- Running without `--sample-ratio` produces the same number of selected raw
  samples as before.
- Running with `--sample-ratio 1.0` keeps all matched samples.
- Running with `--sample-ratio 0.25` selects a deterministic subset before
  train/val splitting.
- The same seed, matched samples, and ratio produce the same selected sample
  set.
- Different seeds can produce a different selected sample set.
- Reordering the input sample list does not change the selected sample set for
  the same seed and ratio.
- Invalid ratios fail clearly:
  - `0`
  - negative values
  - values greater than `1`
- The standalone script exposes `--sample-ratio`.
- `auto-label-tool create-dataset` exposes `--sample-ratio` or reads the same
  behavior from config.
- `auto-label-tool run` passes the same override through to the dataset step if
  CLI overrides are added.
- Config-driven dataset creation supports `dataset.sample_ratio`.
- Tests cover deterministic selection independently from label writing.
- Tests prove sampling happens before train/val splitting.
- Existing `bbox` default behavior remains backward compatible.
- Existing `mask` label output remains unchanged except for the smaller selected
  raw sample set when sampling is requested.
- Python syntax checks pass for touched scripts.

## Engineering Notes

The selected sample set must be based on stable identities, not on whatever
order `Path.rglob()` or set iteration happens to return. A reproducible random
seed is meaningless if the input order changes between machines.

This PR should keep the current train/val split API unless a tiny internal
change is needed to test ordering. Do not use PR3 as an excuse to implement the
separate flexible split feature.

The current pipeline links images before writing labels. Be careful not to make
partial output worse. If label generation skips invalid masks, the PR must not
claim that sampling solved image-without-label behavior unless it actually
changes that behavior and tests it. Prefer documenting that existing limitation
as out of scope.

## Review Risks

- A fake deterministic implementation that still depends on filesystem order.
- Sampling after split, which changes train/val proportions incorrectly and does
  not satisfy the issue.
- Updating only `create_training_dataset_from_masks.py` while leaving
  `auto-label-tool create-dataset` unchanged.
- Overloading this PR with sample counts, flexible splits, or copy/symlink mode.
- Tests that assert one hard-coded shuffled order without proving the underlying
  requirement.

## Suggested Branch

`feat/deterministic-dataset-subsampling`

## Suggested PR Title

`feat: add deterministic dataset subsampling`

## Suggested Checks

```bash
python3 -m py_compile \
  auto_labeling_tool/app/create_training_dataset_from_masks.py \
  auto_labeling_tool/app/cli.py \
  auto_labeling_tool/config.py
```

```bash
python3 -m pytest tests/test_create_training_dataset_from_masks.py
```

```bash
git diff --check
```
