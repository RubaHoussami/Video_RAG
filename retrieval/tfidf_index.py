from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval.utils import SearchHit
from retrieval.base_index import BaseIndex


class TFIDFIndex(BaseIndex):
    """
    TF-IDF index for text retrieval.
    """

    def __init__(self, media_name: str) -> None:
        self._SAVE_PATH = Path(f"./data/tfidf/{media_name}/index.joblib")

        if self._SAVE_PATH.exists():
            self._docs, self._id_map, self.vectorizer, self.tfidf_matrix = joblib.load(self._SAVE_PATH)
        else:
            self._docs = []
            self._id_map = []
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def add(self, docs: List[str], ids: Sequence[str]) -> None:
        if len(docs) != len(ids):
            raise ValueError("docs and ids must match length")

        self._docs.extend(docs)
        self._id_map.extend(ids)
        self.tfidf_matrix = self.vectorizer.fit_transform(self._docs)

    def query(self, queries: List[str], k: int = 5) -> List[List[SearchHit]]:
        if not self._docs or self.tfidf_matrix is None:
            return [[] for _ in queries]

        query_matrix = self.vectorizer.transform(queries)
        sims = cosine_similarity(query_matrix, self.tfidf_matrix)

        results = []
        for row in sims:
            if row.max() <= 0:
                results.append([])
                continue

            row = row / row.max()
            top_idx = np.argsort(row)[::-1][:k]
            hits = [SearchHit(self._id_map[i], float(row[i])) for i in top_idx if row[i] > 0]
            results.append(hits)
        return results

    def save(self) -> None:
        self._SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._docs, self._id_map, self.vectorizer, self.tfidf_matrix), self._SAVE_PATH)
