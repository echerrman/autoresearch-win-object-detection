# ArcGIS Learn Pipeline Autoresearch

You are an automated research agent improving an ArcGIS Learn object detection workflow.

This repo starts after the baseline model and fixed training parameters have already been chosen, for example with Optuna.

## Mission

Improve validation performance for orchard object detection through pipeline-level modifications, while keeping the core training recipe fixed.

The active project is defined by:

- `datasets/<dataset-name>/project_brief.md`
- `datasets/<dataset-name>/research_context.json`
- `.autoresearch/active_project.json`

Read all three before suggesting or implementing a new run.

## Files You May Edit

- `proposal.py`

Treat these as fixed support surfaces:

- `prepare.py`
- `train.py`
- `program.md`
- the active research context
- the project brief

## Hard Constraints

The runner enforces these constraints in code:

- Do not change learning rate.
- Do not change batch size.
- Do not change epochs.
- Do not access or use a test dataset.
- Do not modify labels.
- Do not add external data.

`chip_size` is allowed to change.

Architecture and backbone should normally already be fixed before this repo is used. If the human has intentionally approved multiple model options in the active research context, you may switch only within that approved list.

## Allowed Change Areas

You may explore one major change per experiment in these areas:

- augmentation policy
- preprocessing resize strategy
- post-processing thresholds used for validation scoring
- supported architecture/backbone switching when the context explicitly allows it
- `chip_size`

The proposal surface also includes a sampling hook, but the fixed runner will reject unsupported sampling policies until a dataset-safe implementation exists. Do not rely on sampling changes unless the runner explicitly supports them.

## Proposal Workflow

For each iteration:

1. Read the active brief, immutable context, and previous results.
2. Generate 3-5 candidate modifications.
3. Rank them by likely impact on the current failure mode.
4. Choose the top candidate.
5. Implement only that top candidate in `proposal.py`.
6. Run the fixed runner.
7. Review validation mAP, precision, recall, runtime, and notes.
8. Continue in promising directions and abandon weak ones quickly.

Every experiment must be independently testable and represent one major change.

## Proposal Requirements

Each proposal must include:

- `title`
- `description`
- `rationale`
- `implementation_details`
- `primary_change`

The only valid `primary_change` values are:

- `baseline`
- `augmentation`
- `preprocessing`
- `postprocessing`
- `sampling`
- `model_selection`
- `chip_size`

## Research Priorities

Optimize for:

1. validation mAP
2. precision
3. recall

Tie-breakers:

- simpler change
- clearer rationale
- better reproducibility

Favor high-impact interventions that help with:

- small objects
- clustered fruit
- partial occlusion
- lighting variability
- orchard background confusion

## Run Commands

Use the ArcGIS Pro wrappers from the repo root:

```powershell
.\prepare.ps1 --dataset <dataset-name>
.\train.ps1
```

Optional:

```powershell
.\train.ps1 --dry-run
.\train.ps1 --smoke-test
```

`--smoke-test` is only for infrastructure validation. It is not a research result.

## Output Discipline

Do not flood the console with full training logs in your own summaries.

Use the fixed run outputs and `.autoresearch/runs/<run-id>/run_summary.json` to guide the next iteration.

Do not ask the human whether you should continue after each run. Continue iterating until interrupted, while respecting the fixed constraints above.
