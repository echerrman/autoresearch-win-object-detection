"""
Fixed preflight and dataset/context validation for ArcGIS Learn autoresearch.

Run this file through ArcGIS Pro Python:
    .\\prepare.ps1 --dataset <dataset-name>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
STATE_DIR = REPO_ROOT / ".autoresearch"
ACTIVE_PROJECT_PATH = STATE_DIR / "active_project.json"
ENVIRONMENT_REPORT_PATH = STATE_DIR / "environment_report.json"
RESULTS_PATH = REPO_ROOT / "results.tsv"
RESULTS_HEADER = (
    "run_id\ttimestamp\tproposal_title\tprimary_change\tarchitecture\tbackbone\tchip_size\t"
    "val_map\tval_precision\tval_recall\ttraining_minutes\tstatus\tnotes"
)

KNOWN_EXPORT_MARKERS = (
    "images",
    "labels",
    "map.txt",
    "stats.json",
    "esri_accumulated_stats.json",
    "esri_model_definition.emd",
)
SUPPORTED_CHANGE_AREAS = {
    "augmentation",
    "preprocessing",
    "postprocessing",
    "sampling",
    "model_selection",
    "chip_size",
}
SUPPORTED_ARCHITECTURES = {
    "FasterRCNN",
    "RetinaNet",
    "MaskRCNN",
    "RTDetrV2",
    "YOLOv3",
}


@dataclass
class EnvironmentReport:
    python_executable: str
    python_version: str
    platform: str
    propy_path: str | None
    arcgis_available: bool
    arcgis_version: str | None
    arcgis_learn_available: bool
    arcpy_available: bool
    arcgis_pro_version: str | None
    image_analyst_status: str | None
    torch_version: str | None
    torchvision_version: str | None
    gpu_available: bool | None
    gpu_name: str | None


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def detect_propy_path() -> Path | None:
    candidates = [
        Path("C:/Program Files/ArcGIS/Pro/bin/Python/Scripts/propy.bat"),
        Path.home() / "AppData/Local/Programs/ArcGIS/Pro/bin/Python/Scripts/propy.bat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def inspect_environment() -> EnvironmentReport:
    propy_path = detect_propy_path()
    arcgis_available = False
    arcgis_version = None
    arcgis_learn_available = False
    arcpy_available = False
    arcgis_pro_version = None
    image_analyst_status = None
    torch_version = None
    torchvision_version = None
    gpu_available = None
    gpu_name = None

    try:
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        import arcgis  # type: ignore

        arcgis_available = True
        arcgis_version = getattr(arcgis, "__version__", None)
        try:
            import arcgis.learn  # type: ignore  # noqa: F401

            arcgis_learn_available = True
        except Exception:
            arcgis_learn_available = False
    except Exception:
        arcgis_available = False

    try:
        import arcpy  # type: ignore

        arcpy_available = True
        try:
            arcgis_pro_version = arcpy.GetInstallInfo().get("Version")
        except Exception:
            arcgis_pro_version = None
        try:
            image_analyst_status = arcpy.CheckExtension("ImageAnalyst")
        except Exception:
            image_analyst_status = None
    except Exception:
        arcpy_available = False

    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", None)
        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
        try:
            import torchvision  # type: ignore

            torchvision_version = getattr(torchvision, "__version__", None)
        except Exception:
            torchvision_version = None
    except Exception:
        torch_version = None

    return EnvironmentReport(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        propy_path=str(propy_path) if propy_path else None,
        arcgis_available=arcgis_available,
        arcgis_version=arcgis_version,
        arcgis_learn_available=arcgis_learn_available,
        arcpy_available=arcpy_available,
        arcgis_pro_version=arcgis_pro_version,
        image_analyst_status=image_analyst_status,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


def write_environment_report(report: EnvironmentReport) -> None:
    ensure_state_dir()
    ENVIRONMENT_REPORT_PATH.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def resolve_dataset_dir(dataset: str) -> Path:
    candidate = Path(dataset)
    if not candidate.is_absolute():
        candidate = DATASETS_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"Dataset workspace does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Dataset workspace must be a directory: {candidate}")
    return candidate


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def reject_test_references(value: Any, parent_key: str = "context") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if "test" in key.lower():
                raise ValueError(
                    "Test dataset references are not allowed in research_context.json. "
                    f"Remove the key '{parent_key}.{key}'."
                )
            reject_test_references(nested, f"{parent_key}.{key}")
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            reject_test_references(nested, f"{parent_key}[{idx}]")


def validate_export_workspace(train_export_path: Path) -> list[str]:
    if not train_export_path.exists():
        raise ValueError(f"Exported training workspace is missing: {train_export_path}")
    if not train_export_path.is_dir():
        raise ValueError(f"Exported training workspace must be a directory: {train_export_path}")

    markers = [name for name in KNOWN_EXPORT_MARKERS if (train_export_path / name).exists()]
    if not markers:
        visible = sorted(item.name for item in train_export_path.iterdir())
        preview = ", ".join(visible[:10]) if visible else "<empty>"
        raise ValueError(
            "The exported training workspace does not look like an ArcGIS export. "
            f"Expected one of {KNOWN_EXPORT_MARKERS}, found: {preview}"
        )
    return markers


def normalize_context(context: dict[str, Any], dataset_dir: Path, context_path: Path) -> dict[str, Any]:
    reject_test_references(context)

    framework = context.get("framework")
    if framework != "arcgis.learn":
        raise ValueError("research_context.json must set framework to 'arcgis.learn'.")

    brief_path_raw = context.get("project_brief_path")
    if not isinstance(brief_path_raw, str) or not brief_path_raw.strip():
        raise ValueError("research_context.json must define project_brief_path.")
    project_brief_path = resolve_path(dataset_dir, brief_path_raw)
    if not project_brief_path.exists():
        raise ValueError(f"Project brief does not exist: {project_brief_path}")

    dataset_section = context.get("dataset")
    if not isinstance(dataset_section, dict):
        raise ValueError("research_context.json must define a dataset object.")
    export_path_raw = dataset_section.get("train_export_path")
    if not isinstance(export_path_raw, str) or not export_path_raw.strip():
        raise ValueError("dataset.train_export_path must be defined.")
    train_export_path = resolve_path(dataset_dir, export_path_raw)
    markers = validate_export_workspace(train_export_path)

    baseline_model = context.get("baseline_model")
    if not isinstance(baseline_model, dict):
        raise ValueError("research_context.json must define baseline_model.")
    architecture = baseline_model.get("architecture")
    backbone = baseline_model.get("backbone")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"baseline_model.architecture must be one of {sorted(SUPPORTED_ARCHITECTURES)}."
        )
    if backbone is not None and not isinstance(backbone, str):
        raise ValueError("baseline_model.backbone must be a string or null.")

    approved_models = context.get("approved_models")
    if not isinstance(approved_models, dict) or not approved_models:
        raise ValueError("research_context.json must define approved_models.")
    for model_name, backbones in approved_models.items():
        if model_name not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"approved_models contains unsupported architecture '{model_name}'."
            )
        if backbones is not None and not isinstance(backbones, list):
            raise ValueError(
                f"approved_models.{model_name} must be a list of backbones or null."
            )
        if isinstance(backbones, list) and not all(isinstance(item, str) for item in backbones):
            raise ValueError(
                f"approved_models.{model_name} must contain only backbone strings."
            )

    approved_backbones = approved_models.get(architecture)
    if isinstance(approved_backbones, list) and backbone not in approved_backbones:
        raise ValueError(
            f"baseline backbone '{backbone}' is not allowed for architecture '{architecture}'."
        )

    fixed_parameters = context.get("fixed_parameters")
    if not isinstance(fixed_parameters, dict):
        raise ValueError("research_context.json must define fixed_parameters.")
    for required_key in ("learning_rate", "batch_size", "epochs", "validation_split"):
        if required_key not in fixed_parameters:
            raise ValueError(f"fixed_parameters.{required_key} must be defined.")
    if fixed_parameters["learning_rate"] is None:
        raise ValueError("fixed_parameters.learning_rate must not be null.")
    if not isinstance(fixed_parameters["batch_size"], int) or fixed_parameters["batch_size"] <= 0:
        raise ValueError("fixed_parameters.batch_size must be a positive integer.")
    if not isinstance(fixed_parameters["epochs"], int) or fixed_parameters["epochs"] <= 0:
        raise ValueError("fixed_parameters.epochs must be a positive integer.")
    validation_split = fixed_parameters["validation_split"]
    if not isinstance(validation_split, (int, float)) or not (0.0 < float(validation_split) < 1.0):
        raise ValueError("fixed_parameters.validation_split must be between 0 and 1.")

    baseline_pipeline = context.get("baseline_pipeline")
    if not isinstance(baseline_pipeline, dict):
        raise ValueError("research_context.json must define baseline_pipeline.")
    chip_size = baseline_pipeline.get("chip_size")
    if not isinstance(chip_size, int) or chip_size <= 0:
        raise ValueError("baseline_pipeline.chip_size must be a positive integer.")

    current_best = context.get("current_best_metrics")
    if not isinstance(current_best, dict):
        raise ValueError("research_context.json must define current_best_metrics.")
    for key in ("map", "precision", "recall"):
        if key not in current_best:
            raise ValueError(f"current_best_metrics.{key} must be defined.")

    allowed_change_areas = context.get("allowed_change_areas")
    if not isinstance(allowed_change_areas, list) or not allowed_change_areas:
        raise ValueError("research_context.json must define allowed_change_areas.")
    invalid_areas = [area for area in allowed_change_areas if area not in SUPPORTED_CHANGE_AREAS]
    if invalid_areas:
        raise ValueError(
            f"allowed_change_areas contains unsupported values: {invalid_areas}."
        )
    if "chip_size" not in allowed_change_areas:
        raise ValueError("allowed_change_areas must include 'chip_size'.")

    prohibited_actions = context.get("prohibited_actions")
    if not isinstance(prohibited_actions, list) or not prohibited_actions:
        raise ValueError("research_context.json must define prohibited_actions.")

    normalized = json.loads(json.dumps(context))
    normalized["project_brief_path"] = str(project_brief_path)
    normalized["dataset"]["train_export_path"] = str(train_export_path)
    normalized["dataset_dir"] = str(dataset_dir)
    normalized["research_context_path"] = str(context_path.resolve())
    normalized["export_markers"] = markers
    return normalized


def load_and_validate_context(dataset_dir: Path) -> dict[str, Any]:
    context_path = dataset_dir / "research_context.json"
    if not context_path.exists():
        raise ValueError(f"Missing research_context.json in {dataset_dir}")
    context = load_json(context_path)
    return normalize_context(context, dataset_dir, context_path)


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
        return
    current = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if not current:
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def write_active_project(context: dict[str, Any]) -> dict[str, Any]:
    ensure_state_dir()
    manifest = {
        "dataset_name": Path(context["dataset_dir"]).name,
        "dataset_dir": context["dataset_dir"],
        "train_export_path": context["dataset"]["train_export_path"],
        "project_brief_path": context["project_brief_path"],
        "research_context_path": context["research_context_path"],
        "framework": context["framework"],
        "prepared_at": utc_now_iso(),
    }
    ACTIVE_PROJECT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_active_project() -> dict[str, Any]:
    if not ACTIVE_PROJECT_PATH.exists():
        raise ValueError(
            "No active project is configured. Run prepare.py or .\\prepare.ps1 first."
        )
    return load_json(ACTIVE_PROJECT_PATH)


def run_doctor() -> EnvironmentReport:
    report = inspect_environment()
    write_environment_report(report)
    return report


def format_environment_report(report: EnvironmentReport) -> str:
    lines = [
        f"Python executable: {report.python_executable}",
        f"Python version:    {report.python_version}",
        f"Platform:          {report.platform}",
        f"propy.bat:         {report.propy_path or 'not found'}",
        f"ArcGIS available:  {report.arcgis_available}",
        f"arcgis.learn:      {report.arcgis_learn_available}",
        f"arcpy available:   {report.arcpy_available}",
        f"ArcGIS Pro:        {report.arcgis_pro_version or 'unknown'}",
        f"Image Analyst:     {report.image_analyst_status or 'unknown'}",
        f"torch:             {report.torch_version or 'not found'}",
        f"torchvision:       {report.torchvision_version or 'not found'}",
        f"GPU available:     {report.gpu_available}",
        f"GPU name:          {report.gpu_name or 'n/a'}",
        f"Env report:        {ENVIRONMENT_REPORT_PATH}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ArcGIS Learn autoresearch project")
    parser.add_argument("--dataset", help="Dataset workspace name under datasets/ or an absolute path.")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Only inspect the ArcGIS environment and write the environment report.",
    )
    args = parser.parse_args()

    try:
        report = run_doctor()
        print(format_environment_report(report))
        if args.doctor and not args.dataset:
            return 0

        if not report.arcgis_learn_available:
            raise ValueError(
                "arcgis.learn is not available in the current Python runtime. "
                "Use .\\prepare.ps1 or ArcGIS Pro's propy.bat."
            )
        if report.image_analyst_status not in (None, "Available"):
            raise ValueError(
                f"Image Analyst is not available: {report.image_analyst_status}"
            )
        if not args.dataset:
            raise ValueError("--dataset is required unless --doctor is used by itself.")

        dataset_dir = resolve_dataset_dir(args.dataset)
        context = load_and_validate_context(dataset_dir)
        manifest = write_active_project(context)
        ensure_results_file()

        print()
        print(f"Dataset workspace: {dataset_dir}")
        print(f"Project brief:     {context['project_brief_path']}")
        print(f"Train export:      {context['dataset']['train_export_path']}")
        print(f"Active project:    {ACTIVE_PROJECT_PATH}")
        print(f"Results log:       {RESULTS_PATH}")
        print(f"Prepared at:       {manifest['prepared_at']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
