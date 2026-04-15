# Dataset Workspaces

Create one folder per research project:

```text
datasets/<dataset-name>/
  project_brief.md
  research_context.json
  train_export/
```

- `project_brief.md` is the human-authored description of the imagery, target object, and failure modes.
- `research_context.json` is the immutable machine-readable context for the fixed runner.
- `train_export/` contains the output of ArcGIS Pro's `Export Training Data For Deep Learning` tool.

Normal workflow:

1. Use Optuna outside this repo to choose the baseline model and fixed training parameters.
2. Put those fixed choices into `research_context.json`.
3. Use this repo for constrained agent-driven pipeline iteration.

Use `datasets/example-project/` as a template.
