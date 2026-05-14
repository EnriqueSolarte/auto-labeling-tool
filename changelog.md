# Changelog

## 1) SAM2 propagation and dataset pipeline foundation (Dec 3-9, 2025)
- **Started repository and core SAM2 workflows** from the initial commit through propagation improvements.
- **Key commits:**
  - `bc1094b` (2025-12-03 16:24:50 +0800) - Init commit
  - `a9620f5` (2025-12-04 02:52:29 +0000) - Stable SAM2 propagation through sequences
  - `af2e3f2` (2025-12-04 03:00:39 +0000) - Interactive SAM2 video propagation with recursive prompts
  - `c996da4` (2025-12-04 07:40:01 +0000) - YOLO training dataset creation/visualization scripts
  - `af271f6` (2025-12-09 07:38:38 +0000) - cleanup/refactor of dataset and mask handling

## 2) Environment setup and analysis tooling expansion (Jan 8-19, 2026)
- **Operationalized the project** with Docker + requirements and improved pipeline documentation.
- **Added reporting and visualization utilities** to support dataset quality checks.
- **Key commits:**
  - `0f64486` (2026-01-08 06:07:59 +0000) - Docker setup files + requirements
  - `643477f` (2026-01-08 06:08:28 +0000) - documentation and mask processing improvements
  - `2070eda` (2026-01-14 00:59:02 +0000) and `e0517cd` (2026-01-15 02:46:21 +0000) - mask aggregation improvements
  - `b0e47a9` (2026-01-15 02:58:22 +0000) - sequence dataset visualization utility
  - `58784f3` (2026-01-19 06:31:27 +0000) - dataset statistics utility

## 3) Annotation validation and dataset maintenance workflows (Feb 3-4, 2026)
- **Introduced validation-first dataset maintenance** for annotations and split visualization.
- **Added practical curation operations** like pruning blacklisted samples and ensuring missing label files are created.
- **Key commits:**
  - `fcacddc` (2026-02-03 03:13:16 +0000) - annotation validation tool
  - `d9ac5e5` (2026-02-03 06:49:39 +0000) - visualization controls improvements
  - `8c6fd8e` (2026-02-04 02:04:06 +0000) - dataset pruning mode
  - `5f4e153` (2026-02-04 05:09:26 +0000) - ensure-labels mode
  - `18a29a0` (2026-02-04 05:16:09 +0000) and `5a326c2` (2026-02-04 06:32:58 +0000) - docs refresh and test-split visualization
