from typing import List, Sequence, Literal

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

from retrieval.utils import SearchHit
from config import config
from retrieval.base_index import BaseIndex
from retrieval.utils import l2_normalize


class PGVectorIndex(BaseIndex):
    """
    PostgreSQL + pgvector index for vector similarity search.
    """

    def __init__(self, table: str, dim: int, index_type: Literal["ivfflat", "hnsw", "flat"], index_params: dict = None) -> None:
        self.dim = dim
        self.table = table
        self.index_type = index_type
        self.index_params = index_params or {}
        self.conn = psycopg2.connect(config.PGVECTOR_CONN_STR)
        self.conn.autocommit = True
        self._ensure_schema()

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must match length")

        vecs = l2_normalize(vectors.astype(np.float32, copy=False))
        with self.conn.cursor() as cur:
            sql = (
                f"INSERT INTO {self.table} (id, embedding) "
                "VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
            )
            rows = [(doc_id, vec.tolist()) for doc_id, vec in zip(ids, vecs)]
            execute_batch(cur, sql, rows, page_size=1000)

    def query(self, queries: np.ndarray, k: int = 5) -> List[List[SearchHit]]:
        qvecs = l2_normalize(queries.astype(np.float32, copy=False))
        all_hits = []

        with self.conn.cursor() as cur:
            for vec in qvecs:
                cur.execute(
                    f"""
                    SELECT id, 1 - (embedding <=> %s) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (vec.tolist(), vec.tolist(), k),
                )
                hits = [SearchHit(row[0], float(row[1])) for row in cur.fetchall()]
                all_hits.append(hits)

        return all_hits

    def _ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({self.dim})
                )
                """
            )

            match self.index_type:
                case "ivfflat":
                    params = self.index_params or {"lists": 100}
                    param_str = ", ".join(f"{k} = {v}" for k, v in params.items())
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.table}_ivfflat
                        ON {self.table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH ({param_str})
                        """
                    )

                case "hnsw":
                    params = self.index_params or {"m": 16, "ef_construction": 200}
                    param_str = ", ".join(f"{k} = {v}" for k, v in params.items())
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.table}_hnsw
                        ON {self.table}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH ({param_str})
                        """
                    )

                case "flat":
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.table}_flat
                        ON {self.table}
                        USING flat (embedding vector_cosine_ops)
                        """
                    )

    def save(self) -> None:
        pass
