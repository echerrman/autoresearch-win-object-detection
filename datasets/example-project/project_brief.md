# Project Brief

## Detection Objective

Detect peaches in high-resolution orchard imagery.

## Scene Context

- Imagery source: aerial or drone imagery
- Viewpoint: top-down or near nadir
- Orchard structure: clustered fruit, leaves, branches, mixed lighting

## Likely Challenges

- peaches are small relative to the chip
- fruit may be partially occluded by leaves
- clustered fruit can trigger missed detections or merged boxes
- sun angle and canopy shadows can affect contrast

## Validation Priorities

1. improve validation mAP
2. maintain useful precision
3. avoid harming recall on clustered or partially occluded fruit

## Notes For The Agent

- The baseline model and fixed training parameters were chosen before this repo is used, ideally with Optuna.
- prefer simple, high-impact pipeline changes
- keep experiments independently testable
- do not touch fixed training knobs from the research context
