"""
The AI agent should edit only this file during normal experimentation.

This repo assumes the baseline model and fixed training parameters were chosen
beforehand, for example with Optuna.

Each proposal must represent one major change.
Valid primary_change values:
    - baseline
    - augmentation
    - preprocessing
    - postprocessing
    - chip_size

Forbidden changes are enforced by the fixed training runner:
    - learning_rate
    - batch_size
    - epochs
    - baseline model selection
    - test dataset access
"""

PROPOSAL = {
    "title": "Baseline control",
    "description": "Run the fixed baseline without pipeline changes.",
    "rationale": (
        "The first run should establish a clean baseline with the user's chosen model "
        "and fixed parameters before the agent begins trying improvements."
    ),
    "implementation_details": (
        "No augmentation, preprocessing, postprocessing, or chip-size overrides are "
        "enabled in this baseline proposal."
    ),
    "primary_change": "baseline",
    "augmentation": None,
    "preprocessing": None,
    "postprocessing": None,
    "chip_size_override": None,
}
