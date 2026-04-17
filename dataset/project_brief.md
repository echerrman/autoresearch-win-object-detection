# Project Brief

This file is here to help the AI agent understand your dataset better.

You do not need to rewrite all of it. If you want the fastest possible setup, you can leave most of this file alone and just replace the obvious example details.

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

- The baseline model and fixed training parameters were chosen before this repo is used.
- Prefer simple, high-impact pipeline changes.
- Keep experiments independently testable.
- Do not touch the fixed model or fixed training parameters from `project_config.json`.
