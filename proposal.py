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
    - sampling
    - model_selection
    - chip_size

Forbidden changes are enforced by train.py:
    - learning_rate
    - batch_size
    - epochs
    - test dataset access
"""

PROPOSAL = {
    "title": "Baseline control",
    "description": "Run the approved baseline without pipeline changes.",
    "rationale": (
        "Every research loop needs a fresh control so validation improvements can be "
        "attributed to one explicit pipeline change."
    ),
    "implementation_details": (
        "No augmentation, preprocessing, postprocessing, model-selection, sampling, "
        "or chip-size overrides are enabled in this baseline proposal."
    ),
    "primary_change": "baseline",
    "augmentation": None,
    "preprocessing": None,
    "postprocessing": None,
    "sampling": None,
    "model_selection": None,
    "chip_size_override": None,
}
