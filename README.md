# ArcGIS Learn Pipeline Autoresearch

Use an AI coding agent to iterate on an `arcgis.learn` object detection pipeline after you have already chosen the baseline model and fixed training parameters.

This repo is for the second half of the workflow:

1. Use Optuna outside this repo to find the model and fixed training recipe you want to start from.
2. Put that fixed baseline into this repo.
3. Let an AI agent iterate on constrained pipeline changes one experiment at a time.

The target use case is peach detection in orchard imagery, but the structure is reusable for other ArcGIS Learn object detection projects.

## What This Repo Does

This repo is a guarded research harness for `arcgis.learn`.

It is designed to help an AI agent test pipeline-level changes such as:

- augmentation policy
- preprocessing and resize strategy
- post-processing thresholds used for validation scoring
- `chip_size`

It is not the place where you should broadly search learning rate, batch size, epochs, or baseline model choice. The intended workflow is to choose those first with Optuna, then lock them here.

## What You Need Before You Start

Bring these with you:

- ArcGIS Pro on Windows
- ArcGIS Pro deep learning dependencies installed
- access to ArcGIS Pro Python through `propy.bat`
- an exported training dataset from `Export Training Data For Deep Learning`
- a baseline model choice and fixed training recipe, ideally selected beforehand with Optuna

At minimum, you should already know:

- which model family/backbone you want to start from
- the fixed learning rate
- the fixed batch size
- the fixed epoch count
- the validation split

## How It Works

The repo stays intentionally small:

- [prepare.py](./prepare.py)
  Validates the ArcGIS environment, your dataset workspace, and the immutable research context.
- [train.py](./train.py)
  Runs one fixed `arcgis.learn` training experiment and logs the validation result.
- [proposal.py](./proposal.py)
  The only file the AI agent should edit during normal research.
- [program.md](./program.md)
  The instruction file you point your coding agent at.

The human sets the guardrails. The agent proposes and tests one major change at a time inside those guardrails.

## Project Layout

```text
prepare.py
train.py
proposal.py
program.md
results.tsv
datasets/<dataset-name>/
  project_brief.md
  research_context.json
  train_export/
```

- `project_brief.md` is the plain-English description of the detection problem.
- `research_context.json` is the machine-readable locked context for the run.
- `train_export/` holds the ArcGIS export workspace.

Run artifacts are written to `.autoresearch/` and ignored by git.

## Quick Start

### 1. Choose the baseline outside this repo

Use Optuna to determine the model and fixed parameters you want to bring into this project.

The normal expectation is:

- Optuna chooses the baseline model family/backbone.
- Optuna chooses learning rate, batch size, and epochs.
- This repo treats those choices as fixed.

If you want the AI agent locked to one model only, set `approved_models` in the research context to just that one Optuna-selected option.

### 2. Clone the repo

Open **PowerShell** on Windows, then run:

```powershell
git clone https://github.com/echerrman/autoresearch-win-object-detection.git
cd autoresearch-win-object-detection
```

From this point on, run the rest of the commands from the repo root in **PowerShell** unless stated otherwise.

### 3. Create a dataset workspace

In **PowerShell** or in File Explorer, copy [datasets/example-project](./datasets/example-project/) and rename it to your project name.

```text
datasets/<dataset-name>/
  project_brief.md
  research_context.json
  train_export/
```

For our example peach detecting project:

```text
datasets/peach-orchard-spring-2026/
  project_brief.md
  research_context.json
  train_export/
```

PowerShell example:

```powershell
Copy-Item -Recurse .\datasets\example-project .\datasets\peach-orchard-spring-2026
```

### 4. Drop in your ArcGIS export

In **ArcGIS Pro**, run `Export Training Data For Deep Learning` as you normally would. After it finishes, copy the exported training-data workspace into your project folder.

Target folder in this repo:

```text
datasets/<dataset-name>/train_export/
```

For our example peach detecting project:

```text
datasets/peach-orchard-spring-2026/train_export/
```

If your ArcGIS export already contains folders and files such as `images`, `labels`, and metadata files, copy those contents into `train_export/`.

### 5. Fill in the two project files

Open these two files in your editor:

- [datasets/example-project/project_brief.md](./datasets/example-project/project_brief.md)
- [datasets/example-project/research_context.json](./datasets/example-project/research_context.json)

For your real project, that usually means editing:

```text
datasets/<dataset-name>/project_brief.md
datasets/<dataset-name>/research_context.json
```

For our example peach detecting project:

```text
datasets/peach-orchard-spring-2026/project_brief.md
datasets/peach-orchard-spring-2026/research_context.json
```

`project_brief.md` should explain the real-world context, such as:

- what you are detecting
- where the imagery came from
- what makes the task hard
- whether misses or false positives are more costly

For our example peach detecting project:

```md
# Project Brief

## Detection Objective

Detect peaches in high-resolution drone imagery of a commercial orchard.

## Scene Context

- Imagery source: drone orthomosaic
- Viewpoint: top-down / near nadir
- Target object: peaches
- Common issues: leaf occlusion, clustered fruit, shadows, bright sunlit canopy

## Validation Priorities

1. improve validation mAP
2. keep useful precision
3. avoid missing partially occluded peaches
```

`research_context.json` should lock in:

- the Optuna-selected model family/backbone
- the fixed learning rate
- the fixed batch size
- the fixed epochs
- the validation split
- the baseline `chip_size`
- the current best validation metrics, if known

For our example peach detecting project:

```json
{
  "project_name": "peach-orchard-spring-2026",
  "framework": "arcgis.learn",
  "project_brief_path": "project_brief.md",
  "dataset": {
    "train_export_path": "train_export",
    "dataset_type": "PASCAL_VOC_rectangles"
  },
  "baseline_model": {
    "architecture": "FasterRCNN",
    "backbone": "resnet50",
    "pretrained_path": null
  },
  "approved_models": {
    "FasterRCNN": [
      "resnet50"
    ]
  },
  "fixed_parameters": {
    "learning_rate": 0.001,
    "batch_size": 4,
    "epochs": 20,
    "validation_split": 0.1
  },
  "baseline_pipeline": {
    "chip_size": 320
  },
  "current_best_metrics": {
    "map": 0.42,
    "precision": 0.71,
    "recall": 0.64
  },
  "allowed_change_areas": [
    "augmentation",
    "preprocessing",
    "postprocessing",
    "chip_size"
  ],
  "prohibited_actions": [
    "Do not change learning_rate",
    "Do not change batch_size",
    "Do not change epochs",
    "Do not access a test dataset",
    "Do not modify labels or add external data"
  ]
}
```

### 6. Check your ArcGIS environment

In **PowerShell** at the repo root, run:

```powershell
.\doctor.ps1
```

This confirms that ArcGIS Pro Python, `arcgis.learn`, and the required dependencies are available.

### 7. Validate the project workspace

Still in **PowerShell** at the repo root, run:

```powershell
.\prepare.ps1 --dataset <dataset-name>
```

For our example peach detecting project:

```powershell
.\prepare.ps1 --dataset peach-orchard-spring-2026
```

This validates the dataset folder, the locked context, and the active project selection.

### 8. Dry-run the runner

Still in **PowerShell** at the repo root, run:

```powershell
.\train.ps1 --dry-run
```

This checks that the runner can load your active project and proposal without starting a real training job.

### 9. Run one experiment

Still in **PowerShell** at the repo root, run:

```powershell
.\train.ps1
```

Optional infrastructure-only smoke test:

```powershell
.\train.ps1 --smoke-test
```

`--smoke-test` is only for checking the harness. It is not a research result.

### 10. Hand the repo to your AI agent

Once the setup works, start your coding agent in this repo and point it at [program.md](./program.md).

Example starter prompt:

```text
Read program.md and the active dataset workspace, then start the next constrained research iteration.
Generate 3-5 candidate pipeline changes, choose the strongest one, update proposal.py, run the experiment, and continue within the repo's fixed constraints.
```

For our example peach detecting project:

```text
Read program.md and the active peach-orchard-spring-2026 dataset workspace, then start the next constrained research iteration.
Focus on improving peach detection in top-down orchard drone imagery. Generate 3-5 candidate pipeline changes, choose the strongest one, update proposal.py, run the experiment, and continue within the repo's fixed constraints.
```

## What The Agent Is Allowed To Change

During normal research, the agent should edit only [proposal.py](./proposal.py).

The runner enforces the important constraints in code.

Fixed and normally immutable:

- learning rate
- batch size
- epochs

Allowed research levers:

- augmentation policy
- preprocessing resize strategy
- post-processing thresholds used for validation scoring
- `chip_size`

Model and backbone selection should normally already be chosen before this repo, for example with Optuna. If you intentionally want to permit limited model switching, you can allow that in `approved_models`, but the default recommendation is to keep it narrow.

Explicitly disallowed:

- test dataset access
- label edits
- external data
- arbitrary custom model code

## What You Get Out

Each run produces a reproducible research record:

- one new row in [results.tsv](./results.tsv)
- one run folder under `.autoresearch/runs/<run-id>/`
- snapshots of the active context, proposal, and project brief
- a machine-readable run summary

The main metric is validation `mAP`. Precision and recall are also logged when they are available from the ArcGIS Learn model object.

The intended outcome is not "fully automatic deployment." The intended outcome is a clean record of which constrained pipeline changes improved or harmed validation performance, so you can converge on a stronger ArcGIS Learn workflow for your dataset.

## Current Scope

This repo is deliberately not:

- a replacement for Optuna
- a broad hyperparameter tuning system
- a generic PyTorch model-development workspace
- a place to use the test set during iteration

It is a narrow, reproducible harness for agent-driven pipeline research after the baseline model and fixed training recipe have already been chosen.
