import json
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.faiss_index import FaissIndex
from core.text_embedder import TextEmbedder
from core.image_embedder import ImageEmbedder

with open("evaluation/gold_test_set.json", "r") as f:
    gold = json.load(f)["gold_test_set"]

answerable = gold["answerable"]
unanswerable = gold["unanswerable"]

# === Time Utilities ===
def timestamp_to_seconds(timestamp: str) -> float:
    mins, secs = map(int, timestamp.split(":"))
    return mins * 60 + secs

def is_hit_correct(gold_time: str, retrieved_span: str) -> bool:
    try:
        gold_sec = timestamp_to_seconds(gold_time)
        start, duration = map(float, retrieved_span.split("+"))
        end = start + duration
        return start <= gold_sec <= end
    except:
        return False

# === Evaluation Function ===
def evaluate(index, embed_fn, runs=3) -> dict:
    total_answerable = len(answerable) * runs
    correct = 0
    latencies = []

    for _ in range(runs):
        for q in answerable:
            query_vector = embed_fn([q["question"]])
            start_time = time.time()
            results = index.query(query_vector, k=1)
            end_time = time.time()
            latencies.append(end_time - start_time)

            top_id = results[0][0].id if results and results[0] else None
            if top_id and is_hit_correct(q["answer_timestamp"], top_id):
                correct += 1

    return {
        "accuracy": correct / total_answerable,
        "latency": np.mean(latencies)
    }

# === Initialize FAISS Indexes and Embedders ===
faiss_audio = FaissIndex("text_faiss", 1024)
faiss_video = FaissIndex("image_faiss", 512)

text_embedder = TextEmbedder("e5")
image_embedder = ImageEmbedder("ViT-B-32", "openai")

# === Run Evaluation ===
print("Running evaluation...")

audio_metrics = evaluate(faiss_audio, embed_fn=lambda texts: text_embedder._embed(texts))
video_metrics = evaluate(faiss_video, embed_fn=image_embedder.encode_text)

# === Report Results ===
print("\nEvaluation Results:")
print(f"{'Method':<15} {'Accuracy':>10}{'Avg Latency (s)':>18}")
print("-" * 60)
print(f"{'Audio':<15} {audio_metrics['accuracy']:>10.2f} {audio_metrics['latency']:>18.4f}")
print(f"{'Video':<15} {video_metrics['accuracy']:>10.2f} {video_metrics['latency']:>18.4f}")

# === Plot Results ===
methods = ['Audio', 'Video']
accuracy_vals = [audio_metrics['accuracy'], video_metrics['accuracy']]

plt.figure()
x = np.arange(len(methods))
width = 0.35
plt.bar(x - width/2, accuracy_vals, width, label='Accuracy')
plt.ylabel('Score')
plt.title('Retrieval Accuracy Audio vs. Video')
plt.xticks(x, methods)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
