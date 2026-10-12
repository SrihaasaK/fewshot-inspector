"""fewshot-inspector: training-free few-shot visual defect inspection (DINOv2 / AnomalyDINO)."""

from inspector.backbone import Backbone, BackboneLoadError, ImageLoadError, load_image, select_device
from inspector.bank import PatchBank
from inspector.config import InspectorConfig
from inspector.inspector import InspectionResult, Inspector, Prediction

__all__ = [
    "Backbone",
    "BackboneLoadError",
    "ImageLoadError",
    "InspectionResult",
    "Inspector",
    "InspectorConfig",
    "PatchBank",
    "Prediction",
    "load_image",
    "select_device",
]
__version__ = "0.1.0"
