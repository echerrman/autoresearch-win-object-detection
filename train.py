"""
Fixed arcgis.learn experiment runner.

Run this file through ArcGIS Pro Python:
    .\\train.ps1
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from prepare import (
    RESULTS_HEADER,
    RESULTS_PATH,
    STATE_DIR,
    ensure_state_dir,
    load_active_project,
    load_and_validate_project_config,
)

REPO_ROOT = Path(__file__).resolve().parent
PROPOSAL_PATH = REPO_ROOT / "proposal.py"
RUNS_DIR = STATE_DIR / "runs"
EPSILON = 1e-6
DISALLOWED_PROPOSAL_KEYS = {"lr", "learning_rate", "batch_size", "epochs", "num_epochs"}
SUPPORTED_PRIMARY_CHANGES = {
    "baseline",
    "augmentation",
    "preprocessing",
    "postprocessing",
    "chip_size",
}
ACTIVE_SECTION_MAP = {
    "augmentation": "augmentation",
    "preprocessing": "preprocessing",
    "postprocessing": "postprocessing",
    "chip_size": "chip_size_override",
}
SUPPORTED_AUGMENTATION_KWARGS = {
    "do_flip",
    "flip_vert",
    "max_rotate",
    "max_zoom",
    "max_lighting",
    "max_warp",
    "p_affine",
    "p_lighting",
}
SUPPORTED_PREPROCESSING_KEYS = {"resize_to"}
SUPPORTED_POSTPROCESSING_KEYS = {"detect_thresh", "iou_thresh"}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "experiment"


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
        return
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
        return
    if lines[0] != RESULTS_HEADER:
        raise ValueError(
            "results.tsv has an unexpected header. Archive or replace it before running new ArcGIS Learn experiments."
        )


def load_project_from_active_project(project_override: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    active_project = load_active_project()
    if project_override:
        requested = Path(project_override).resolve()
        active = Path(active_project["project_dir"]).resolve()
        if requested != active:
            raise ValueError(
                "train.py runs against the prepared active project. "
                "Run .\\prepare.ps1 again if you want to switch projects."
            )
    project_config = load_and_validate_project_config(Path(active_project["project_dir"]))
    return active_project, project_config


def load_proposal() -> dict[str, Any]:
    if not PROPOSAL_PATH.exists():
        raise ValueError(f"Missing proposal file: {PROPOSAL_PATH}")
    spec = importlib.util.spec_from_file_location("autoresearch_proposal", PROPOSAL_PATH)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import proposal from {PROPOSAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proposal = getattr(module, "PROPOSAL", None)
    if not isinstance(proposal, dict):
        raise ValueError("proposal.py must expose a top-level PROPOSAL dictionary.")
    return copy.deepcopy(proposal)


def _scan_for_disallowed_keys(value: Any, path: str = "proposal") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in DISALLOWED_PROPOSAL_KEYS:
                raise ValueError(
                    f"Forbidden proposal field '{path}.{key}'. "
                    "learning_rate, batch_size, and epochs are immutable."
                )
            _scan_for_disallowed_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            _scan_for_disallowed_keys(nested, f"{path}[{idx}]")


def _is_active_section(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def validate_proposal(proposal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    required_fields = {
        "title",
        "description",
        "rationale",
        "implementation_details",
        "primary_change",
        "augmentation",
        "preprocessing",
        "postprocessing",
        "chip_size_override",
    }
    missing = sorted(required_fields - set(proposal))
    if missing:
        raise ValueError(f"proposal.py is missing required fields: {missing}")

    _scan_for_disallowed_keys(proposal)

    if _is_active_section(proposal.get("sampling")):
        raise ValueError(
            "Sampling is not part of the simplified user workflow. "
            "Choose augmentation, preprocessing, postprocessing, or chip_size."
        )
    if _is_active_section(proposal.get("model_selection")):
        raise ValueError(
            "The baseline model is fixed in this repo. "
            "Choose your model before you start, then keep model selection out of proposal.py."
        )

    primary_change = proposal["primary_change"]
    if primary_change not in SUPPORTED_PRIMARY_CHANGES:
        raise ValueError(
            f"proposal.primary_change must be one of {sorted(SUPPORTED_PRIMARY_CHANGES)}."
        )

    active_sections = [
        section
        for section, field_name in ACTIVE_SECTION_MAP.items()
        if _is_active_section(proposal.get(field_name))
    ]

    if primary_change == "baseline":
        if active_sections:
            raise ValueError("Baseline proposals must not activate any change sections.")
    else:
        if active_sections != [primary_change]:
            raise ValueError(
                "Each proposal must implement exactly one major change. "
                f"Expected only '{primary_change}' to be active, found {active_sections}."
            )

    allowed_change_areas = set(context.get("allowed_change_areas", []))
    if primary_change != "baseline" and primary_change not in allowed_change_areas:
        raise ValueError(
            f"primary_change '{primary_change}' is not allowed by the active project."
        )

    augmentation = proposal.get("augmentation")
    if augmentation:
        if not isinstance(augmentation, dict):
            raise ValueError("proposal.augmentation must be an object.")
        if augmentation.get("kind", "fastai_get_transforms") != "fastai_get_transforms":
            raise ValueError("Only augmentation kind 'fastai_get_transforms' is supported.")
        kwargs = augmentation.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError("proposal.augmentation.kwargs must be an object.")
        invalid = sorted(set(kwargs) - SUPPORTED_AUGMENTATION_KWARGS)
        if invalid:
            raise ValueError(
                f"Unsupported augmentation kwargs: {invalid}. "
                f"Supported keys are {sorted(SUPPORTED_AUGMENTATION_KWARGS)}."
            )

    preprocessing = proposal.get("preprocessing")
    if preprocessing:
        if not isinstance(preprocessing, dict):
            raise ValueError("proposal.preprocessing must be an object.")
        invalid = sorted(set(preprocessing) - SUPPORTED_PREPROCESSING_KEYS)
        if invalid:
            raise ValueError(
                f"Unsupported preprocessing keys: {invalid}. "
                f"Supported keys are {sorted(SUPPORTED_PREPROCESSING_KEYS)}."
            )

    postprocessing = proposal.get("postprocessing")
    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("proposal.postprocessing must be an object.")
        invalid = sorted(set(postprocessing) - SUPPORTED_POSTPROCESSING_KEYS)
        if invalid:
            raise ValueError(
                f"Unsupported postprocessing keys: {invalid}. "
                f"Supported keys are {sorted(SUPPORTED_POSTPROCESSING_KEYS)}."
            )

    chip_size_override = proposal.get("chip_size_override")
    if chip_size_override is not None:
        if not isinstance(chip_size_override, int) or chip_size_override <= 0:
            raise ValueError("proposal.chip_size_override must be a positive integer or null.")

    return proposal


def build_transforms(augmentation: dict[str, Any] | None):
    if not augmentation:
        return None
    from fastai.vision import get_transforms  # type: ignore

    kwargs = augmentation.get("kwargs", {})
    return get_transforms(**kwargs)


def resolve_runtime_configuration(context: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    model = context["model"]
    baseline_pipeline = context["baseline_pipeline"]
    fixed_parameters = context["fixed_parameters"]

    preprocessing = proposal.get("preprocessing") or {}
    postprocessing = proposal.get("postprocessing") or {}

    return {
        "architecture": model["architecture"],
        "backbone": model.get("backbone"),
        "pretrained_path": model.get("pretrained_path"),
        "chip_size": proposal.get("chip_size_override") or baseline_pipeline["chip_size"],
        "resize_to": preprocessing.get("resize_to", baseline_pipeline.get("resize_to")),
        "detect_thresh": postprocessing.get(
            "detect_thresh", baseline_pipeline.get("detect_thresh", 0.2)
        ),
        "iou_thresh": postprocessing.get(
            "iou_thresh", baseline_pipeline.get("iou_thresh", 0.1)
        ),
        "learning_rate": fixed_parameters["learning_rate"],
        "batch_size": fixed_parameters["batch_size"],
        "epochs": fixed_parameters["epochs"],
        "validation_split": fixed_parameters["validation_split"],
        "fit_kwargs": dict(context.get("fit_parameters", {})),
        "dataset_type": context["dataset"]["dataset_type"],
    }


def import_arcgis_stack():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import arcgis.learn as arcgis_learn  # type: ignore

    return arcgis_learn


def build_data(arcgis_learn, context: dict[str, Any], proposal: dict[str, Any], run_dir: Path, runtime: dict[str, Any]):
    transforms = build_transforms(proposal.get("augmentation"))
    data = arcgis_learn.prepare_data(
        path=context["dataset"]["train_export_path"],
        chip_size=runtime["chip_size"],
        val_split_pct=runtime["validation_split"],
        batch_size=runtime["batch_size"],
        transforms=transforms,
        resize_to=runtime["resize_to"],
        working_dir=str(run_dir),
        dataset_type=runtime["dataset_type"],
    )
    return data


def build_model(arcgis_learn, data, runtime: dict[str, Any]):
    if not hasattr(arcgis_learn, runtime["architecture"]):
        raise ValueError(
            f"arcgis.learn does not expose architecture '{runtime['architecture']}'."
        )
    model_class = getattr(arcgis_learn, runtime["architecture"])
    signature = inspect.signature(model_class)
    kwargs: dict[str, Any] = {}
    if "backbone" in signature.parameters and runtime["backbone"] is not None:
        kwargs["backbone"] = runtime["backbone"]
    if "pretrained_path" in signature.parameters and runtime["pretrained_path"]:
        kwargs["pretrained_path"] = runtime["pretrained_path"]
    return model_class(data, **kwargs)


def extract_precision_recall(model: Any) -> tuple[Any, Any]:
    for candidate in (
        getattr(model, "_model_metrics", None),
        getattr(model, "_get_model_metrics", None),
    ):
        metrics = candidate() if callable(candidate) else candidate
        if isinstance(metrics, dict):
            precision = None
            recall = None
            for key, value in metrics.items():
                lowered = str(key).lower()
                if "precision" in lowered and precision is None:
                    precision = value
                if "recall" in lowered and recall is None:
                    recall = value
            if precision is not None or recall is not None:
                return precision, recall
    return None, None


def read_best_map_from_results() -> float | None:
    if not RESULTS_PATH.exists():
        return None
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        values = []
        for row in reader:
            raw = (row.get("val_map") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if not raw or status == "crash":
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return max(values) if values else None


def append_result(row: dict[str, Any]) -> None:
    ensure_results_file()
    fieldnames = RESULTS_HEADER.split("\t")
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)


def snapshot_run_inputs(
    run_dir: Path, active_project: dict[str, Any], context: dict[str, Any], proposal: dict[str, Any]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "active_project.json").write_text(
        json.dumps(active_project, indent=2), encoding="utf-8"
    )
    (run_dir / "project_config.snapshot.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    (run_dir / "proposal.snapshot.json").write_text(
        json.dumps(proposal, indent=2), encoding="utf-8"
    )
    brief_path = Path(active_project["project_brief_path"])
    if brief_path.exists():
        (run_dir / "project_brief.snapshot.md").write_text(
            brief_path.read_text(encoding="utf-8"), encoding="utf-8"
        )


def choose_status(current_best_map: float | None, val_map: float | None) -> str:
    if val_map is None:
        return "crash"
    if current_best_map is None:
        return "keep"
    return "keep" if val_map > (current_best_map + EPSILON) else "discard"


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    ensure_state_dir()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    project_override = args.project or args.dataset
    active_project, context = load_project_from_active_project(project_override)
    proposal = validate_proposal(load_proposal(), context)
    runtime = resolve_runtime_configuration(context, proposal)

    run_id = args.run_id or (
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{slugify(proposal['title'])}"
    )
    run_dir = RUNS_DIR / run_id
    snapshot_run_inputs(run_dir, active_project, context, proposal)

    if args.dry_run:
        summary = {
            "run_id": run_id,
            "mode": "dry-run",
            "timestamp": utc_now_iso(),
            "project_name": context["project_name"],
            "proposal_title": proposal["title"],
            "primary_change": proposal["primary_change"],
            "architecture": runtime["architecture"],
            "backbone": runtime["backbone"],
            "chip_size": runtime["chip_size"],
            "run_dir": str(run_dir),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    arcgis_learn = import_arcgis_stack()
    t0 = time.time()
    data = build_data(arcgis_learn, context, proposal, run_dir, runtime)
    model = build_model(arcgis_learn, data, runtime)

    fit_kwargs = dict(runtime["fit_kwargs"])
    fit_kwargs.setdefault("monitor", "average_precision")
    fit_kwargs.setdefault("checkpoint", True)

    epochs = runtime["epochs"]
    if args.smoke_test:
        epochs = 1

    model.fit(epochs=epochs, lr=runtime["learning_rate"], **fit_kwargs)
    training_minutes = (time.time() - t0) / 60.0

    val_map = model.average_precision_score(
        detect_thresh=runtime["detect_thresh"],
        iou_thresh=runtime["iou_thresh"],
        mean=True,
        show_progress=True,
    )
    try:
        per_class_map = model.average_precision_score(
            detect_thresh=runtime["detect_thresh"],
            iou_thresh=runtime["iou_thresh"],
            mean=False,
            show_progress=False,
        )
    except Exception:
        per_class_map = None

    precision, recall = extract_precision_recall(model)
    prior_best = read_best_map_from_results()
    status = "smoke" if args.smoke_test else choose_status(prior_best, val_map)
    notes = proposal["description"]

    summary = {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "project_name": context["project_name"],
        "proposal_title": proposal["title"],
        "primary_change": proposal["primary_change"],
        "architecture": runtime["architecture"],
        "backbone": runtime["backbone"],
        "chip_size": runtime["chip_size"],
        "resize_to": runtime["resize_to"],
        "learning_rate": runtime["learning_rate"],
        "batch_size": runtime["batch_size"],
        "epochs_requested": runtime["epochs"],
        "epochs_executed": epochs,
        "detect_thresh": runtime["detect_thresh"],
        "iou_thresh": runtime["iou_thresh"],
        "val_map": val_map,
        "val_precision": precision,
        "val_recall": recall,
        "per_class_map": per_class_map,
        "training_minutes": training_minutes,
        "status": status,
        "notes": notes,
        "run_dir": str(run_dir),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.smoke_test:
        append_result(
            {
                "run_id": run_id,
                "timestamp": summary["timestamp"],
                "proposal_title": proposal["title"],
                "primary_change": proposal["primary_change"],
                "architecture": runtime["architecture"],
                "backbone": runtime["backbone"] or "",
                "chip_size": runtime["chip_size"],
                "val_map": f"{val_map:.6f}" if isinstance(val_map, (int, float)) else "",
                "val_precision": "" if precision is None else precision,
                "val_recall": "" if recall is None else recall,
                "training_minutes": f"{training_minutes:.2f}",
                "status": status,
                "notes": notes,
            }
        )

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "---",
        f"run_id:            {summary.get('run_id')}",
        f"proposal_title:    {summary.get('proposal_title')}",
        f"primary_change:    {summary.get('primary_change')}",
        f"architecture:      {summary.get('architecture', '')}",
        f"backbone:          {summary.get('backbone', '')}",
        f"chip_size:         {summary.get('chip_size', '')}",
        f"val_map:           {summary.get('val_map', '')}",
        f"val_precision:     {summary.get('val_precision', '')}",
        f"val_recall:        {summary.get('val_recall', '')}",
        f"training_minutes:  {summary.get('training_minutes', '')}",
        f"status:            {summary.get('status', '')}",
    ]
    if summary.get("run_dir"):
        lines.append(f"run_dir:           {summary.get('run_dir')}")
    return "\n".join(lines)


def write_crash_summary(
    run_id: str, error: str, context: dict[str, Any] | None, proposal: dict[str, Any] | None
) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    architecture = ""
    backbone = ""
    chip_size = ""
    if context:
        architecture = context.get("model", {}).get("architecture", "")
        backbone = context.get("model", {}).get("backbone", "")
        chip_size = context.get("baseline_pipeline", {}).get("chip_size", "")
    if proposal and proposal.get("chip_size_override") is not None:
        chip_size = proposal["chip_size_override"]

    summary = {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "status": "crash",
        "error": error,
        "proposal_title": proposal.get("title") if proposal else "",
        "primary_change": proposal.get("primary_change") if proposal else "",
        "architecture": architecture,
        "backbone": backbone,
        "chip_size": chip_size,
        "val_map": "",
        "val_precision": "",
        "val_recall": "",
        "training_minutes": "",
        "notes": error,
        "run_dir": str(run_dir),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed ArcGIS Learn autoresearch experiment")
    parser.add_argument(
        "--project",
        help="Optional project folder path. Must match the prepared active project.",
    )
    parser.add_argument(
        "--dataset",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate project and proposal without training.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a one-epoch infrastructure validation. This does not append to results.tsv.",
    )
    parser.add_argument("--run-id", help="Optional explicit run identifier.")
    args = parser.parse_args()

    ensure_state_dir()
    ensure_results_file()

    context = None
    proposal = None
    run_id = args.run_id or f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-startup"
    try:
        _, context = load_project_from_active_project(args.project or args.dataset)
        proposal = load_proposal()
        if args.run_id:
            run_id = args.run_id
        else:
            run_id = (
                f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{slugify(proposal.get('title', 'experiment'))}"
            )
        summary = run_experiment(args)
        print(format_summary(summary))
        return 0
    except Exception as exc:
        summary = write_crash_summary(run_id, str(exc), context, proposal)
        if not args.dry_run and not args.smoke_test:
            append_result(
                {
                    "run_id": summary["run_id"],
                    "timestamp": summary["timestamp"],
                    "proposal_title": summary["proposal_title"],
                    "primary_change": summary["primary_change"],
                    "architecture": summary["architecture"],
                    "backbone": summary["backbone"],
                    "chip_size": summary["chip_size"],
                    "val_map": "",
                    "val_precision": "",
                    "val_recall": "",
                    "training_minutes": "",
                    "status": "crash",
                    "notes": str(exc),
                }
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
