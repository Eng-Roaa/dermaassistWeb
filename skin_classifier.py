"""
skin_classifier.py  -  wraps the existing TFLite model.

Preprocessing is identical to skin_camera_app1.py:
RGB -> resize to 224x224 -> float32 (0-255, no extra normalisation) -> batch of 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

CLASSES = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]
THRESHOLD = 0.35


@dataclass
class Prediction:
    model_name: str
    major_class: str          # top-scoring class (even when uncertain)
    confidence: float         # 0..1
    uncertain: bool           # True when confidence < THRESHOLD
    scores: dict              # {class_name: score 0..1}
    processing_time: float    # seconds: set tensor + invoke + read output


class SkinClassifier:
    def __init__(self, model_path: Path | str, classes=CLASSES, threshold=THRESHOLD):
        self.model_path = Path(model_path)
        self.classes = list(classes)
        self.threshold = threshold

        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict(self, image: Image.Image) -> Prediction:
        resized = image.convert("RGB").resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)[None, ...]

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_index, array)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_index)[0]
        elapsed = time.perf_counter() - start

        if len(scores) != len(self.classes):
            raise ValueError(f"Unexpected number of model outputs: {len(scores)}")

        best = int(np.argmax(scores))
        confidence = float(scores[best])
        return Prediction(
            model_name=self.model_path.name,
            major_class=self.classes[best],
            confidence=confidence,
            uncertain=confidence < self.threshold,
            scores={name: float(s) for name, s in zip(self.classes, scores)},
            processing_time=elapsed,
        )
