from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .feature_selection import HybridFeatureSelector


torch.manual_seed(42)


class SpamClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(x_tensor)


@dataclass
class TrainingArtifacts:
    vectorizer: TfidfVectorizer
    selector: HybridFeatureSelector
    label_encoder: LabelEncoder
    model: SpamClassifier


class SpamPipeline:
    def __init__(self, max_features: int = 5000, k_features: int = 2000):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        self.selector = HybridFeatureSelector(k_features=k_features)
        self.label_encoder = LabelEncoder()
        self.model: SpamClassifier | None = None

    def train(self, texts: List[str], labels: List[str], epochs: int = 12) -> Dict:
        y_encoded = self.label_encoder.fit_transform(labels)
        x_vectors = self.vectorizer.fit_transform(texts)
        x_selected = self.selector.fit_transform(x_vectors, y_encoded)
        x_tensor = torch.tensor(x_selected.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self.model = SpamClassifier(x_tensor.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return {
            "classes": self.label_encoder.classes_.tolist(),
            "features": int(x_tensor.shape[1]),
            "selector": {
                "k_features": self.selector.k_features,
                "chi2_weight": self.selector.chi2_weight,
                "mi_weight": self.selector.mi_weight,
            },
            "epochs": epochs,
        }

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        x_vectors = self.vectorizer.transform(texts)
        x_selected = self.selector.transform(x_vectors)
        x_tensor = torch.tensor(x_selected.toarray(), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(x_tensor).squeeze(1).numpy()
        results = []
        for prob in probabilities:
            label = self.label_encoder.inverse_transform([int(prob >= 0.5)])[0]
            results.append({"label": label, "spam_probability": float(prob)})
        return results
