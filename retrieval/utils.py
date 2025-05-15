from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(slots=True)
class SearchHit:
    id: str
    score: float

def tokenize(text: str) -> List[str]:
    return text.lower().split()

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)