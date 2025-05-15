from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from retrieval.utils import SearchHit
from retrieval.base_index import BaseIndex
from retrieval.utils import l2_normalize


class FaissIndex(BaseIndex):
    """
    Flat inner-product index (cosine if vectors are unit-norm).
    """

    def __init__(self, media_name: str, dim: int) -> None:
        self._SAVE_DIR = Path(f"./data/faiss/{media_name}")
        self._INDEX_FILE = self._SAVE_DIR / "index.faiss"
        self._IDS_FILE = self._SAVE_DIR / "index.ids.npy"

        self._dim = dim

        if self._INDEX_FILE.exists() and self._IDS_FILE.exists():
            self.index = faiss.read_index(str(self._INDEX_FILE))
            self._id_map = np.load(self._IDS_FILE).tolist()
        else:
            self.index = faiss.IndexFlatIP(dim)
            self._id_map = []

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must match length")

        vectors = l2_normalize(vectors.astype(np.float32))
        self.index.add(vectors)
        self._id_map.extend(ids)

    def query(self, queries: np.ndarray, k: int = 5) -> List[List[SearchHit]]:
        queries = l2_normalize(queries.astype(np.float32))
        scores, idx = self.index.search(queries, k)
        hits = []
        for row_scores, row_idx in zip(scores, idx):
            row = [SearchHit(self._id_map[i], float(s)) for i, s in zip(row_idx, row_scores) if i != -1]
            hits.append(row)
        return hits

    def save(self) -> None:
        self._SAVE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._INDEX_FILE))
        np.save(self._IDS_FILE, np.asarray(self._id_map))
