# ArcGIS Learn Pipeline Autoresearch

Use an AI coding agent to iterate on an `arcgis.learn` object detection pipeline after you have already chosen the baseline model and fixed training parameters.

It is built around:

- [runtime/prepare.py](./runtime/prepare.py) for environment checks and project prep
- [runtime/train.py](./runtime/train.py) for the fixed training runner
- [proposal.py](./proposal.py) as the only file the AI agent should edit during normal research
- [program.md](./program.md) as the instruction file you point your AI agent at

The target use case is peach detection in orchard imagery, but the structure is reusable for other ArcGIS Learn object detection projects.

## What This Repo Does

This repo is for the second half of a object detection model training workflow:

1. Use Optuna outside this repo to decide the baseline model and fixed training recipe.
2. Put those fixed choices into this repo.
3. Run one baseline training experiment with no changes.
4. Let an AI agent iterate on constrained pipeline improvements one change at a time.

This repo is **not** where you broadly tune learning rate, batch size, epochs, or model selection. It assumes those decisions were already made before you got here.

## What You Need Before You Start

Bring these with you:

- ArcGIS Pro on Windows
- ArcGIS Pro deep learning dependencies installed
- access to ArcGIS Pro Python through `propy.bat`
- an exported training dataset from `Export Training Data For Deep Learning`
- a baseline model choice and fixed training recipe, ideally selected beforehand with Optuna

Before you start using this repo, Optuna should ideally already have decided:

- the model architecture
- the backbone
- the learning rate
- the batch size
- the epoch count
- the starting chip size
- optionally the validation split if you do not want to use the repo default of 10%

## Supported Model Values

Use the exact `architecture` and `backbone` format below in `dataset/project_config.json`.

| `architecture` value | `backbone` value | Notes |
| --- | --- | --- |
| `FasterRCNN` | ResNet family, such as `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152` | Good default two-stage detector. |
| `RetinaNet` | ResNet family, such as `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152` | Single-stage detector with ResNet-family backbone. |
| `MaskRCNN` | ResNet family, such as `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152` | Supported here as a fixed model choice, though it is an instance-segmentation architecture. |
| `RTDetrV2` | ResNet family, such as `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152` | ArcGIS docs describe a ResNet-family backbone here. |
| `YOLOv3` | `null` | In this repo, `YOLOv3` uses its built-in backbone. Do not supply a separate backbone string. |

Examples:

```json
"model": {
  "architecture": "FasterRCNN",
  "backbone": "resnet50"
}
```

```json
"model": {
  "architecture": "YOLOv3",
  "backbone": null
}
```

## Getting Started

For normal setup, the user only needs to touch:

- [dataset/project_brief.md](./dataset/project_brief.md)
- [dataset/project_config.json](./dataset/project_config.json)
- [dataset/train_export/](./dataset/train_export/)

`project_brief.md` is a plain-English note for the AI agent. It is helpful, but you do not need to rewrite all of it.

`project_config.json` is the only JSON file the user needs to edit.

## Quick Start

I will use an example use case of detecting Peaches to help explain the steps.

### 1. Choose the baseline outside this repo

Use Optuna to determine the fixed baseline you want to bring into this project.

At minimum, decide:

- the model architecture
- the backbone
- the learning rate
- the batch size
- the epoch count
- the starting chip size

Optional but supported:

- a non-default validation split
- a pretrained checkpoint path

For our example peach detecting project, Optuna might have already told you:

- architecture: `FasterRCNN`
- backbone: `resnet50`
- learning rate: `0.001`
- batch size: `4`
- epochs: `20`
- starting chip size: `320`

### 2. Clone the repo

Open **PowerShell** on Windows, then run:

```powershell
git clone https://github.com/echerrman/autoresearch-win-object-detection.git
cd autoresearch-win-object-detection
```

From this point on, run the rest of the commands from the repo root in **PowerShell** unless stated otherwise.

### 3. Use the built-in `dataset/` folder

This repo assumes one dataset per clone.

You do **not** need to copy or rename any template folder. Just use the built-in `dataset/` folder that already comes with the repo:

```text
dataset/
  project_brief.md
  project_config.json
  train_export/
```

### 4. Copy your ArcGIS export into `dataset/train_export/`

In **ArcGIS Pro**, run `Export Training Data For Deep Learning` as you normally would.

That ArcGIS tool will create an export folder. Inside that export folder, you will usually see things like:

- an `images` folder
- a `labels` folder
- utility files such as `map.txt`, `stats.json`, `esri_accumulated_stats.json`, or `.emd` files

Copy the **contents** of that ArcGIS export folder into this repo folder:

```text
dataset/train_export/
```

Do **not** copy the outer ArcGIS export folder itself as an extra nested folder.

For our example peach detecting project, after copying, it should look roughly like:

```text
dataset/train_export/
  images/
  labels/
  map.txt
  stats.json
```

### 5. Fill in `project_brief.md` and `project_config.json`

Open these two files in your editor:

- [dataset/project_brief.md](./dataset/project_brief.md)
- [dataset/project_config.json](./dataset/project_config.json)

#### `project_brief.md`

This file is mainly for helping the AI agent understand the dataset better.

You do not have to rewrite everything. If you want the easiest startup, just replace the obvious example details.

For our example peach detecting project, the brief might say things like:

- the target is peaches
- the imagery is top-down drone imagery
- fruit can be small, clustered, and partially occluded
- misses are especially bad when peaches are hidden by leaves

#### `project_config.json`

This file holds the fixed baseline choices that the AI agent must not change.

If you are using `YOLOv3`, set `"backbone": null`. For the ResNet-based models, use a ResNet-family backbone string such as `resnet50`.

The default template is intentionally short:

```json
{
  "project_name": "peach-orchard-example",
  "model": {
    "architecture": "FasterRCNN",
    "backbone": "resnet50"
  },
  "fixed_parameters": {
    "learning_rate": 0.001,
    "batch_size": 4,
    "epochs": 20
  },
  "chip_size": 320
}
```

If you want to override defaults later, you can also add optional fields such as:

- `validation_split`
- `pretrained_path`

### 6. Check your ArcGIS environment

Still in **PowerShell** at the repo root, run:

```powershell
.\doctor.ps1
```

This checks that ArcGIS Pro Python, `arcgis.learn`, `arcpy`, and the required dependencies are available.

This step may take around **20 to 40 seconds** on a normal machine because it launches ArcGIS Pro Python and imports the deep learning stack.

### 7. Prepare the project

Still in **PowerShell** at the repo root, run:

```powershell
.\prepare.ps1
```

This validates:

- the ArcGIS environment
- `dataset/project_config.json`
- `dataset/project_brief.md`
- `dataset/train_export/`

If everything looks good, it writes the active-project state used by the training runner.

### 8. Dry-run the runner

Still in **PowerShell** at the repo root, run:

```powershell
.\train.ps1 --dry-run
```

This checks that the runner can load the active project and the current proposal without starting a real training job.

### 9. Run one baseline experiment

Still in **PowerShell** at the repo root, run:

```powershell
.\train.ps1
```

This first run should be the plain baseline run using your fixed model and fixed parameters.

In other words, before the AI tries to improve anything, let the repo run one experiment with:

- your fixed architecture
- your fixed backbone
- your fixed learning rate
- your fixed batch size
- your fixed epochs
- your starting chip size

That gives you the baseline result for this repo's own validation workflow.

Optional infrastructure-only smoke test:

```powershell
.\train.ps1 --smoke-test
```

`--smoke-test` is only for checking the harness. It is not a research result.

### 10. Hand the repo to your AI agent

Once the setup works, start your coding agent in this repo and point it at [program.md](./program.md).

Example starter prompt:

```text
Read program.md, dataset/project_brief.md, and dataset/project_config.json, then start the next constrained research iteration.
Generate 3-5 candidate pipeline changes, choose the strongest one, update proposal.py, run the experiment, and continue within the repo's fixed constraints.
```

For our example peach detecting project:

```text
Read program.md, dataset/project_brief.md, and dataset/project_config.json, then start the next constrained research iteration.
This project is for detecting peaches in top-down orchard drone imagery. Generate 3-5 candidate pipeline changes, choose the strongest one, update proposal.py, run the experiment, and continue within the repo's fixed constraints.
```

If you want the agent to stop at a certain point, say that directly in your prompt. For example:

- `Stop after 10 iterations.`
- `Stop after 5 hours.`
- `Stop if validation precision reaches 0.85.`
- `Stop if you have three consecutive non-improving experiments.`

## What The Agent Is Allowed To Change

During normal research, the agent should edit only [proposal.py](./proposal.py).

The runner enforces the important constraints in code.

Fixed and immutable:

- model architecture
- backbone
- learning rate
- batch size
- epochs

Allowed research levers:

- augmentation policy
- preprocessing resize strategy
- post-processing thresholds used for validation scoring
- `chip_size`

Explicitly disallowed:

- test dataset access
- label edits
- external data
- arbitrary custom model code

## What You Get Out

Each run produces a reproducible research record:

- one new row in [results.tsv](./results.tsv)
- one run folder under `.autoresearch/runs/<run-id>/`
- snapshots of the active config, proposal, and brief
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
