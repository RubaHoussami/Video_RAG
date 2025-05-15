from typing import List
from retrieval.utils import SearchHit


class HitReranker:
    """
    Combines and reranks results from multiple modalities (e.g., text and image).
    """

    @staticmethod
    def normalize_scores(hits: List[SearchHit]) -> List[SearchHit]:
        if not hits:
            return []

        max_score = max(hit.score for hit in hits)
        if max_score == 0:
            return hits

        return [SearchHit(id=hit.id, score=hit.score / max_score) for hit in hits]

    @staticmethod
    def merge(text_hits: List[SearchHit], image_hits: List[SearchHit], text_weight: float = 0.5) -> List[SearchHit]:
        norm_text = HitReranker.normalize_scores(text_hits)
        norm_image = HitReranker.normalize_scores(image_hits)

        text_map = {hit.id: hit.score for hit in norm_text}
        image_map = {hit.id: hit.score for hit in norm_image}

        all_ids = set(text_map.keys()) | set(image_map.keys())
        merged_hits = []

        for id_ in all_ids:
            score = text_weight * text_map.get(id_, 0.0) + (1 - text_weight) * image_map.get(id_, 0.0)
            merged_hits.append(SearchHit(id=id_, score=score))

        return sorted(merged_hits, key=lambda h: h.score, reverse=True)
