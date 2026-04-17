"""
Fixed preflight and project validation for ArcGIS Learn autoresearch.

Run this file through ArcGIS Pro Python:
    .\\doctor.ps1
    .\\prepare.ps1
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
DEFAULT_PROJECT_DIR = REPO_ROOT / "dataset"
PROJECT_BRIEF_FILENAME = "project_brief.md"
PROJECT_CONFIG_FILENAME = "project_config.json"
LEGACY_PROJECT_CONFIG_FILENAME = "research_context.json"
TRAIN_EXPORT_DIRNAME = "train_export"

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
MODEL_CONFIG_RULES = {
    "FasterRCNN": {
        "backbone_policy": "resnet_family",
        "default_backbone": "resnet50",
    },
    "RetinaNet": {
        "backbone_policy": "resnet_family",
        "default_backbone": "resnet50",
    },
    "MaskRCNN": {
        "backbone_policy": "resnet_family",
        "default_backbone": "resnet50",
    },
    "RTDetrV2": {
        "backbone_policy": "resnet_family",
        "default_backbone": "resnet18",
    },
    "YOLOv3": {
        "backbone_policy": "fixed_internal",
        "default_backbone": None,
    },
}
SUPPORTED_ARCHITECTURES = set(MODEL_CONFIG_RULES)
SUPPORTED_RESNET_BACKBONES = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
}
DEFAULT_ALLOWED_CHANGE_AREAS = [
    "augmentation",
    "preprocessing",
    "postprocessing",
    "chip_size",
]
DEFAULT_PROHIBITED_ACTIONS = [
    "Do not change learning_rate",
    "Do not change batch_size",
    "Do not change epochs",
    "Do not access a test dataset",
    "Do not modify labels or add external data",
    "Do not change the fixed baseline model",
]
DEFAULT_FIT_PARAMETERS = {
    "one_cycle": True,
    "early_stopping": False,
    "checkpoint": True,
    "tensorboard": False,
    "mixed_precision": False,
    "monitor": "average_precision",
}
DEFAULT_DATASET_TYPE = "PASCAL_VOC_rectangles"
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_DETECT_THRESH = 0.2
DEFAULT_IOU_THRESH = 0.1


def normalize_architecture_name(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError("model.architecture must be a string.")
    collapsed = "".join(ch for ch in raw_value.lower() if ch.isalnum())
    aliases = {
        "fasterrcnn": "FasterRCNN",
        "retinanet": "RetinaNet",
        "maskrcnn": "MaskRCNN",
        "rtdetrv2": "RTDetrV2",
        "yolov3": "YOLOv3",
    }
    return aliases.get(collapsed)


def normalize_backbone_name(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError("model.backbone must be a string or null.")
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    collapsed = "".join(ch for ch in cleaned.lower() if ch.isalnum())
    if collapsed in SUPPORTED_RESNET_BACKBONES:
        return collapsed
    return cleaned


def validate_model_choice(architecture: str, backbone: str | None) -> tuple[str, str | None]:
    rule = MODEL_CONFIG_RULES[architecture]
    backbone_policy = rule["backbone_policy"]

    if backbone_policy == "fixed_internal":
        if backbone is not None:
            raise ValueError(
                f"{architecture} uses its built-in backbone in this repo. "
                "Set model.backbone to null."
            )
        return architecture, None

    if backbone is None:
        raise ValueError(
            f"model.backbone is required for {architecture}. "
            f"Use one of {sorted(SUPPORTED_RESNET_BACKBONES)}."
        )

    if backbone not in SUPPORTED_RESNET_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}' for {architecture}. "
            f"Use one of {sorted(SUPPORTED_RESNET_BACKBONES)}."
        )

    return architecture, backbone


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


def resolve_project_dir(project: str | None) -> Path:
    if not project:
        candidate = DEFAULT_PROJECT_DIR.resolve()
        if not candidate.exists():
            raise ValueError(
                f"Default project folder does not exist: {candidate}. "
                "Clone the repo again or recreate the dataset template."
            )
        return candidate

    raw_path = Path(project)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (REPO_ROOT / project).resolve(),
                (REPO_ROOT / "datasets" / project).resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError(f"Project workspace must be a directory: {candidate}")
            return candidate

    raise ValueError(
        f"Project workspace does not exist: {project}. "
        f"By default this repo expects a folder at {DEFAULT_PROJECT_DIR}."
    )


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


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
            "The train_export folder does not look like an ArcGIS export. "
            "Copy the contents of the ArcGIS export into dataset/train_export/. "
            f"Expected one of {KNOWN_EXPORT_MARKERS}, found: {preview}"
        )
    return markers


def _coerce_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def normalize_project_config(
    raw_config: dict[str, Any], project_dir: Path, config_path: Path
) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path.name} must contain a JSON object.")

    project_name = raw_config.get("project_name") or project_dir.name
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("project_name must be a non-empty string when provided.")

    brief_path_raw = raw_config.get("project_brief_path", PROJECT_BRIEF_FILENAME)
    if not isinstance(brief_path_raw, str) or not brief_path_raw.strip():
        raise ValueError("project_brief_path must be a string when provided.")
    project_brief_path = resolve_path(project_dir, brief_path_raw)
    if not project_brief_path.exists():
        raise ValueError(f"Project brief does not exist: {project_brief_path}")

    export_path_raw = raw_config.get("train_export_path")
    if export_path_raw is None and isinstance(raw_config.get("dataset"), dict):
        export_path_raw = raw_config["dataset"].get("train_export_path")
    if export_path_raw is None:
        export_path_raw = TRAIN_EXPORT_DIRNAME
    if not isinstance(export_path_raw, str) or not export_path_raw.strip():
        raise ValueError("train_export_path must be a string when provided.")
    train_export_path = resolve_path(project_dir, export_path_raw)
    markers = validate_export_workspace(train_export_path)

    model = raw_config.get("model") or raw_config.get("baseline_model")
    if not isinstance(model, dict):
        raise ValueError(
            f"{config_path.name} must define a 'model' object with architecture and backbone."
        )
    raw_architecture = model.get("architecture")
    raw_backbone = model.get("backbone")
    pretrained_path = model.get("pretrained_path")

    architecture = normalize_architecture_name(raw_architecture)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "model.architecture must be one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)}. "
            f"Received: {raw_architecture!r}"
        )
    backbone = normalize_backbone_name(raw_backbone)
    architecture, backbone = validate_model_choice(architecture, backbone)
    if pretrained_path is not None and not isinstance(pretrained_path, str):
        raise ValueError("model.pretrained_path must be a string or null.")
    resolved_pretrained_path = None
    if isinstance(pretrained_path, str) and pretrained_path.strip():
        resolved_pretrained_path = resolve_path(project_dir, pretrained_path)
        if not resolved_pretrained_path.exists():
            raise ValueError(f"model.pretrained_path does not exist: {resolved_pretrained_path}")

    fixed_parameters = raw_config.get("fixed_parameters")
    if not isinstance(fixed_parameters, dict):
        raise ValueError(
            f"{config_path.name} must define a 'fixed_parameters' object."
        )

    learning_rate = fixed_parameters.get("learning_rate")
    if learning_rate is None:
        raise ValueError("fixed_parameters.learning_rate must be defined.")
    learning_rate = _coerce_float(learning_rate, "fixed_parameters.learning_rate")

    batch_size = fixed_parameters.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("fixed_parameters.batch_size must be a positive integer.")

    epochs = fixed_parameters.get("epochs")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("fixed_parameters.epochs must be a positive integer.")

    validation_split = fixed_parameters.get("validation_split", DEFAULT_VALIDATION_SPLIT)
    validation_split = _coerce_float(validation_split, "fixed_parameters.validation_split")
    if not (0.0 < validation_split < 1.0):
        raise ValueError("fixed_parameters.validation_split must be between 0 and 1.")

    chip_size = raw_config.get("chip_size")
    if chip_size is None and isinstance(raw_config.get("baseline_pipeline"), dict):
        chip_size = raw_config["baseline_pipeline"].get("chip_size")
    if not isinstance(chip_size, int) or chip_size <= 0:
        raise ValueError(
            "chip_size must be a positive integer. "
            "This is the starting chip size that the baseline run will use."
        )

    dataset_type = raw_config.get("dataset_type")
    if dataset_type is None and isinstance(raw_config.get("dataset"), dict):
        dataset_type = raw_config["dataset"].get("dataset_type")
    dataset_type = dataset_type or DEFAULT_DATASET_TYPE
    if not isinstance(dataset_type, str) or not dataset_type.strip():
        raise ValueError("dataset_type must be a non-empty string when provided.")

    resize_to = raw_config.get("resize_to")
    if resize_to is None and isinstance(raw_config.get("baseline_pipeline"), dict):
        resize_to = raw_config["baseline_pipeline"].get("resize_to")
    if resize_to is not None and (not isinstance(resize_to, int) or resize_to <= 0):
        raise ValueError("resize_to must be a positive integer or null.")

    detect_thresh = raw_config.get("detect_thresh")
    if detect_thresh is None and isinstance(raw_config.get("baseline_pipeline"), dict):
        detect_thresh = raw_config["baseline_pipeline"].get("detect_thresh")
    detect_thresh = DEFAULT_DETECT_THRESH if detect_thresh is None else _coerce_float(
        detect_thresh, "detect_thresh"
    )

    iou_thresh = raw_config.get("iou_thresh")
    if iou_thresh is None and isinstance(raw_config.get("baseline_pipeline"), dict):
        iou_thresh = raw_config["baseline_pipeline"].get("iou_thresh")
    iou_thresh = DEFAULT_IOU_THRESH if iou_thresh is None else _coerce_float(
        iou_thresh, "iou_thresh"
    )

    current_best = raw_config.get("current_best_metrics") or {}
    current_best_map = current_best.get("map", 0.0)
    if current_best_map is None:
        current_best_map = 0.0
    current_best_map = _coerce_float(current_best_map, "current_best_metrics.map")
    current_best_precision = current_best.get("precision")
    current_best_recall = current_best.get("recall")

    normalized = {
        "project_name": project_name.strip(),
        "framework": "arcgis.learn",
        "project_dir": str(project_dir),
        "project_brief_path": str(project_brief_path),
        "project_config_path": str(config_path.resolve()),
        "dataset": {
            "train_export_path": str(train_export_path),
            "dataset_type": dataset_type,
        },
        "model": {
            "architecture": architecture,
            "backbone": backbone,
            "pretrained_path": str(resolved_pretrained_path) if resolved_pretrained_path else None,
        },
        "fixed_parameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "validation_split": validation_split,
        },
        "fit_parameters": dict(DEFAULT_FIT_PARAMETERS),
        "baseline_pipeline": {
            "chip_size": chip_size,
            "resize_to": resize_to,
            "detect_thresh": detect_thresh,
            "iou_thresh": iou_thresh,
        },
        "current_best_metrics": {
            "map": current_best_map,
            "precision": current_best_precision,
            "recall": current_best_recall,
        },
        "allowed_change_areas": list(DEFAULT_ALLOWED_CHANGE_AREAS),
        "prohibited_actions": list(DEFAULT_PROHIBITED_ACTIONS),
        "export_markers": markers,
    }
    return normalized


def load_and_validate_project_config(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / PROJECT_CONFIG_FILENAME
    if not config_path.exists():
        legacy_path = project_dir / LEGACY_PROJECT_CONFIG_FILENAME
        if legacy_path.exists():
            config_path = legacy_path
        else:
            raise ValueError(
                f"Missing {PROJECT_CONFIG_FILENAME} in {project_dir}. "
                "The easiest setup is to edit dataset/project_config.json."
            )
    raw_config = load_json(config_path)
    return normalize_project_config(raw_config, project_dir, config_path)


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
        return
    current = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if not current:
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def write_active_project(config: dict[str, Any]) -> dict[str, Any]:
    ensure_state_dir()
    manifest = {
        "project_name": config["project_name"],
        "project_dir": config["project_dir"],
        "project_brief_path": config["project_brief_path"],
        "project_config_path": config["project_config_path"],
        "train_export_path": config["dataset"]["train_export_path"],
        "framework": config["framework"],
        "prepared_at": utc_now_iso(),
    }
    ACTIVE_PROJECT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_active_project() -> dict[str, Any]:
    if not ACTIVE_PROJECT_PATH.exists():
        raise ValueError(
            "No active project is configured. Run .\\prepare.ps1 first."
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
    parser.add_argument(
        "--project",
        help="Optional project folder path. Defaults to .\\dataset.",
    )
    parser.add_argument(
        "--dataset",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Only inspect the ArcGIS environment and write the environment report.",
    )
    args = parser.parse_args()

    project_arg = args.project or args.dataset

    try:
        report = run_doctor()
        print(format_environment_report(report))
        if args.doctor and not project_arg:
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

        project_dir = resolve_project_dir(project_arg)
        config = load_and_validate_project_config(project_dir)
        manifest = write_active_project(config)
        ensure_results_file()

        print()
        print(f"Project name:      {config['project_name']}")
        print(f"Project folder:    {project_dir}")
        print(f"Project brief:     {config['project_brief_path']}")
        print(f"Project config:    {config['project_config_path']}")
        print(f"Train export:      {config['dataset']['train_export_path']}")
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
