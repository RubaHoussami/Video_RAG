from pathlib import Path
from typing import Tuple
import json

from core.audio_transcriber import AudioTranscriber
from core.hit_reranker import HitReranker
from core.image_embedder import ImageEmbedder
from core.keyframe_extractor import KeyframeExtractor
from core.media_loader import MediaLoader
from core.text_embedder import TextEmbedder

from retrieval.bm25_index import BM25Index
from retrieval.faiss_index import FaissIndex
from retrieval.pgvector_index import PGVectorIndex
from retrieval.tfidf_index import TFIDFIndex

class Driver:
    """
    Orchestrates processing of media and unified retrieval via selected indices.
    """

    def __init__(self) -> None:
        self.state_file = Path("assets/driver/processed.json")
        self.registry = self._load_registry()
        self.media_loader = MediaLoader()

    def _get_index_type(self, index_obj):
        if isinstance(index_obj, BM25Index):
            return "bm25"
        elif isinstance(index_obj, TFIDFIndex):
            return "tfidf"
        elif isinstance(index_obj, FaissIndex):
            return "faiss"
        elif isinstance(index_obj, PGVectorIndex):
            return index_obj.index_type
        return None

    def _save_registry(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for hashed, data in self.registry.items():
            entry = {"query": data["query"]}

            if "audio" in data:
                audio = data["audio"]
                audio["index"].save()

                entry["audio"] = {
                    "transcriber": audio.get("transcriber"),
                    "embedder": audio.get("embedder"),
                    "index_type": self._get_index_type(audio.get("index"))
                }

            if "video" in data:
                video = data["video"]
                video["index"].save()

                entry["video"] = {
                    "extractor": video.get("extractor"),
                    "embedder": video.get("embedder"),
                    "index_type": self._get_index_type(video.get("index"))
                }

            serializable[hashed] = entry

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _load_registry(self) -> dict:
        if not self.state_file.exists():
            return {}

        with open(self.state_file, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for hashed, data in registry.items():
            if "audio" in data and "index" not in data["audio"]:
                index_type = data["audio"].get("index_type", "bm25")
                index_name = f"text_{index_type}"

                if index_type in {"bm25", "tfidf"}:
                    cls = BM25Index if index_type == "bm25" else TFIDFIndex
                    data["audio"]["index"] = cls(index_name)
                else:
                    dimension = data["audio"]["embedder"]["dimension"]
                    if index_type == "faiss":
                        data["audio"]["index"] = FaissIndex(index_name, dimension)
                    else:
                        data["audio"]["index"] = PGVectorIndex(index_name, dimension, index_type)

            if "video" in data and "index" not in data["video"]:
                index_type = data["video"].get("index_type", "faiss")
                index_name = f"image_{index_type}"
                dimension = data["video"]["embedder"]["dimension"]

                if index_type == "faiss":
                    data["video"]["index"] = FaissIndex(index_name, dimension)
                else:
                    data["video"]["index"] = PGVectorIndex(index_name, dimension, index_type)

        return registry

    def process_new_media(self, url: str = None, uploaded_file = None, configuration: dict = None, attempts: int = 3) -> dict:
        if not url and not uploaded_file:
            return {"error": "Either URL or uploaded file must be provided."}
        elif url and uploaded_file:
            return {"error": "Provide either a URL or an uploaded file, not both."}

        media = self.media_loader.download(url) if url else self.media_loader.upload(uploaded_file)

        if media.hashed in self.registry:
            return {"message": "Media processed successfully.", "hash": media.hashed}

        query_config = {
            "k": configuration.get("k", 5),
            "min_score": configuration.get("min_score", 0.5),
            "text_weight": configuration.get("text_weight", 0.5),
            "epsilon": configuration.get("epsilon", 0.1),
        }

        self.registry[media.hashed] = {"query": query_config}
        processing_method = configuration.get("processing_method", "both")

        try:
            match processing_method:
                case "audio":
                    self._process_audio(media.hashed, configuration, attempts)
                case "video":
                    self._process_video(media.hashed, configuration, attempts)
                case "both":
                    self._process_audio(media.hashed, configuration, attempts)
                    self._process_video(media.hashed, configuration, attempts)
                case _:
                    raise ValueError("Invalid processing method. Choose 'audio', 'video', or 'both'.")
        except Exception as e:
            self.registry.pop(media.hashed, None)
            self.media_loader.delete(media.hashed)
            return {"error": f"Processig failed {e}"}

        self.media_loader.save_state()
        self._save_registry()
        return {"message": "Media processed successfully.", "hash": media.hashed}

    def _process_audio(self, hashed: str, configuration: dict, attempts: int) -> None:
        whisper_model = configuration["whisper_model"]
        model_size = whisper_model.get("model_size", "base")
        min_duration = whisper_model.get("min_duration", 5.0)
        beam_size = whisper_model.get("beam_size", 5)
        temperature = whisper_model.get("temperature", 0.0)

        audio_transcriber = AudioTranscriber(model_size)

        trials = 0
        while trials < attempts:
            try:
                language = audio_transcriber.transcribe(hashed, min_duration, beam_size, temperature)
                index_to_text_map = audio_transcriber.get_mapping(hashed, language)
                break
            except Exception as e:
                trials += 1
                if trials == attempts:
                    raise RuntimeError(f"Audio transcription failed after {attempts} attempts: {e}")

        transcriber_config = {
            "model_size": model_size,
            "min_duration": min_duration,
            "beam_size": beam_size,
            "temperature": temperature,
            "language": language
        }

        text_index = configuration["text_index"]
        index_type = text_index.get("type", "bm25")
        index_name = f"text_{index_type}"

        if index_type in {"bm25", "tfidf"}:
            match index_type:
                case "bm25":
                    index = BM25Index(index_name)
                case "tfidf":
                    index = TFIDFIndex(index_name)

            text_ids = list(index_to_text_map.keys())
            texts = list(index_to_text_map.values())
            try:
                index.add(texts, text_ids)
            except Exception as e:
                raise RuntimeError(f"Text index creation failed: {e}")

            self.registry[hashed].update({
                "audio":{
                    "transcriber": transcriber_config,
                    "index": index
                }
            })
            return

        transformer_model = configuration["transformer_model"]
        model_key = transformer_model.get("model_key", "e5")
        batch_size = transformer_model.get("batch_size", 16)

        embedder = TextEmbedder(model_key, batch_size)

        try:
            embeddings, text_ids = embedder.embed(hashed, language)
            dimension = embeddings.shape[1]
        except Exception as e:
            raise RuntimeError(f"Text embedding failed: {e}")

        embedder_config = {
            "model_key": model_key,
            "batch_size": batch_size,
            "dimension": dimension
        }

        match index_type:
            case "faiss":
                index = FaissIndex(index_name, dimension)
            case "ivfflat" | "hnsw" | "flat":
                index = PGVectorIndex(index_name, dimension, index_type)
            case _:
                raise RuntimeError("Invalid index type. Choose 'bm25', 'faiss', 'ivfflat', 'hnsw', 'flat', or 'tfidf'.")

        index.add(embeddings, text_ids)

        self.registry[hashed].update({
            "audio":{
                "transcriber": transcriber_config,
                "embedder": embedder_config,
                "index": index
            }
        })

    def _process_video(self, hashed: str, configuration: dict, attempts: int = 3) -> None:
        keyframe_extractor = configuration.get("keyframe_extractor", {})
        min_duration = keyframe_extractor.get("min_duration", 5.0)
        scene_threshold = keyframe_extractor.get("scene_threshold", 0.5)
        bins = keyframe_extractor.get("bins", 64)

        extractor = KeyframeExtractor(scene_threshold, bins)

        trials = 0
        while trials < attempts:
            try:
                extractor.extract(hashed, min_duration)
                break
            except Exception as e:
                trials += 1
                if trials == attempts:
                    raise RuntimeError(f"Keyframe extraction failed after {attempts} attempts: {e}")

        extractor_config = {
            "scene_threshold": scene_threshold,
            "bins": bins,
            "min_duration": min_duration
        }

        image_model = configuration["image_model"]
        model_id = image_model.get("type", "ViT-B-32")
        platform = image_model.get("platform", "openai")
        batch_size = image_model.get("batch_size", 16)

        try:
            embedder = ImageEmbedder(model_id, platform, batch_size)
            embeddings, image_ids = embedder.embed(hashed)
            dimension = embeddings.shape[1]
        except Exception as e:
            raise RuntimeError(f"Image embedding failed: {e}")

        embedder_config = {
            "model_id": model_id,
            "platform": platform,
            "batch_size": batch_size,
            "dimension": dimension
        }

        image_index = configuration["image_index"]
        index_type = image_index.get("type", "faiss")
        index_name = f"image_{index_type}"

        match index_type:
            case "faiss":
                index = FaissIndex(index_name, dimension)
            case "ivfflat" | "hnsw" | "flat":
                index = PGVectorIndex(index_name, dimension, index_type)
            case _:
                raise RuntimeError("Invalid index type. Choose 'faiss', 'ivfflat', 'hnsw', or 'flat'.")

        index.add(embeddings, image_ids)

        self.registry[hashed].update({
            "video": {
                "extractor": extractor_config,
                "embedder": embedder_config,
                "index": index
            }
        })

    def delete_media(self, hashed: str) -> bool:
        if hashed in self.registry:
            self.registry.pop(hashed)
            self.media_loader.delete(hashed)
            self._save_registry()
            self.media_loader.save_state()
            return True
        return False

    def list_media(self):
        return self.media_loader.list_media()
    
    def query(self, video_hash: str, query: str) -> dict:
        if video_hash not in self.registry:
            return {"error": "Media not found."}
        
        params = self.registry[video_hash]["query"]
        k = params.get("k", 5)
        min_score = params.get("min_score", 0.5)
        text_weight = params.get("text_weight", 0.5)
        epsilon = params.get("epsilon", 0.1)

        entry = self.registry[video_hash]
        text_hits, image_hits = [], []

        if "audio" in entry:
            audio_data = entry["audio"]
            index = audio_data["index"]
            index_type = self._get_index_type(index)

            if index_type in {"bm25", "tfidf"}:
                text_hits = index.query([query], k=k)[0]
            else:
                embedder_cfg = audio_data["embedder"]
                embedder = TextEmbedder(embedder_cfg["model_key"], embedder_cfg["batch_size"])
                embedded_query = embedder._embed([query])
                text_hits = index.query(embedded_query, k=k)[0]

        if "video" in entry:
            video_data = entry["video"]
            index = video_data["index"]
            embedder_cfg = video_data["embedder"]

            embedder = ImageEmbedder(embedder_cfg["model_id"], embedder_cfg["platform"], embedder_cfg["batch_size"])
            embedded_query = embedder.encode_text([query])
            image_hits = index.query(embedded_query, k=k)[0]

        text_hits = [hit for hit in text_hits if hit.score >= min_score]
        image_hits = [hit for hit in image_hits if hit.score >= min_score]

        if not text_hits and not image_hits:
            return {"message": "No results found."}

        if text_hits and image_hits:
            combined = HitReranker.merge(text_hits, image_hits, alpha=text_weight)
        else:
            combined = text_hits or image_hits

        best_hit = combined[0]
        best_id = best_hit.id
        best_score = best_hit.score

        audio_top = text_hits[0] if text_hits else None
        video_top = image_hits[0] if image_hits else None

        start, end = self._get_span_from_id(best_id)

        if audio_top and video_top:
            score_gap = abs(audio_top.score - video_top.score)
            span_audio = self._get_span_from_id(audio_top.id)
            span_video = self._get_span_from_id(video_top.id)

            if score_gap <= epsilon and self.spans_overlap(span_audio, span_video):
                start = min(span_audio[0], span_video[0])
                end = max(span_audio[1], span_video[1])

        return {
            "id": best_id,
            "score": best_score,
            "start": start,
            "end": end
        }

    def _get_span_from_id(self, segment_id: str) -> Tuple[float, float]:
        start_str, end_str = segment_id.split("+")
        return float(start_str), float(end_str)

    def spans_overlap(self, span1: Tuple[float, float], span2: Tuple[float, float]) -> bool:
        return max(span1[0], span2[0]) < min(span1[1], span2[1])
