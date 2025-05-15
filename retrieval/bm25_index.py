from pathlib import Path
from typing import List, Sequence

import joblib
from rank_bm25 import BM25Okapi
from retrieval.utils import SearchHit, tokenize
from retrieval.base_index import BaseIndex


class BM25Index(BaseIndex):
    """
    BM25 index for text retrieval.
    """

    def __init__(self, media_name: str) -> None:
        self._SAVE_PATH = Path(f"./data/bm25/{media_name}/index.joblib")

        if self._SAVE_PATH.exists():
            self._docs, self._id_map = joblib.load(self._SAVE_PATH)
            self.bm25 = BM25Okapi(self._docs)
        else:
            self._docs = []
            self._id_map = []
            self.bm25 = None

    def add(self, docs: List[str], ids: Sequence[str]) -> None:
        if len(docs) != len(ids):
            raise ValueError("docs and ids must match length")

        tokenized = [tokenize(doc) for doc in docs]
        self._docs.extend(tokenized)
        self._id_map.extend(ids)
        self.bm25 = BM25Okapi(self._docs)

    def query(self, queries: List[str], k: int = 5) -> List[List[SearchHit]]:
        if self.bm25 is None:
            return [[] for _ in queries]

        results = []
        for q in queries:
            scores = self.bm25.get_scores(tokenize(q))
            if not scores.size or scores.max() <= 0:
                results.append([])
                continue

            max_s = float(scores.max())
            top_idx = scores.argsort()[::-1][:k]
            hits = [SearchHit(self._id_map[i], float(scores[i] / max_s)) for i in top_idx if scores[i] > 0]
            results.append(hits)
        return results

    def save(self) -> None:
        self._SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._docs, self._id_map), self._SAVE_PATH)
