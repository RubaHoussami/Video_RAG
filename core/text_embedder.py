from pathlib import Path
from typing import List, Literal, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Multilingual text embedder using top-performing open-source models from the MTEB leaderboards.
    """

    SUPPORTED_MODELS = {
        "e5": "intfloat/multilingual-e5-large-instruct",
        "sfr": "sentence-transformers/sfr-Embedding-Mistral",
        "labse": "sentence-transformers/LaBSE",
        "multi": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "jina": "jinaai/jina-embeddings-v3-base"
    }

    def __init__(self, model_key: Literal["e5", "sfr", "labse", "multi", "jina"] = "e5", batch_size: int = 16) -> None:
        self.model_key = model_key
        self.batch_size = batch_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.SUPPORTED_MODELS[model_key])
        self.model = self.model.to(device)
        self.model.eval()

    def _embed(self, texts: List[str]) -> np.ndarray:
        formatted_texts = [self._format_input(t) for t in texts]
        vectors = []

        for i in range(0, len(formatted_texts), self.batch_size):
            batch = formatted_texts[i : i + self.batch_size]
            vec_batch = self.model.encode(batch, convert_to_numpy=True)
            vectors.append(vec_batch.astype(np.float32))

        return np.vstack(vectors)

    def _format_input(self, text: str) -> str:
        text = text.strip()
        if self.model_key == "e5":
            return f"query: {text}"
        return text

    def embed(self, video_name: str, language: str) -> Tuple[np.ndarray, List[str]]:
        texts_path = Path(f"assets/transcriptions/{video_name}-{language}")
        if not texts_path.exists():
            raise FileNotFoundError(f"Transcript directory does not exist: {texts_path}")

        texts = []
        text_ids = []

        for file in sorted(texts_path.glob("*.txt")):
            with open(file, "r", encoding="utf-8") as f:
                texts.append(f.read())
                text_ids.append(file.stem)

        embeddings = self._embed(texts)
        return embeddings, text_ids
