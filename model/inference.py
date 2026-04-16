from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def _resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


@dataclass
class InferenceResult:
    detected_labels: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]


class InventoryClassifier:
    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        self.device = torch.device("cpu")
        self._transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model, self.class_names, self.thresholds = self._load_model_and_metadata()

    def _load_model_and_metadata(self) -> tuple[torch.nn.Module, list[str], dict[str, float]]:
        artifact = torch.load(self.artifact_path, map_location=self.device)
        if "model_state_dict" not in artifact:
            raise ValueError("Artifact is missing 'model_state_dict'.")
        if "class_names" not in artifact:
            raise ValueError("Artifact is missing 'class_names'.")

        class_names: list[str] = list(artifact["class_names"])
        best_thresholds_raw = artifact.get("best_thresholds", {})
        thresholds = {name: float(best_thresholds_raw.get(name, 0.5)) for name in class_names}

        model = _resnet18(num_classes=len(class_names))
        model.load_state_dict(artifact["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, class_names, thresholds

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict_from_pil(self, image: Image.Image) -> InferenceResult:
        image = image.convert("RGB")
        tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()

        probabilities = {name: float(p) for name, p in zip(self.class_names, probs)}
        detected_labels = [
            name
            for name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            if prob >= self.thresholds.get(name, 0.5)
        ]

        return InferenceResult(
            detected_labels=detected_labels,
            probabilities=probabilities,
            thresholds=self.thresholds,
        )
