# Video RAG System

**Video-RAG** (Retrieval-Augmented Generation) is a multimodal video question-answering system that lets users ask questions about a video and get relevant answers based on both spoken content and visual elements. The system combines **audio transcripts** and **video keyframes**, using a hybrid retrieval mechanism and a modular architecture for flexibility and extensibility.

Streamlit powers the interactive frontend, and the backend supports pluggable retrievers (BM25, FAISS, TF-IDF, pgvector), making this system ideal for both research and real-world deployment.

---

## 📁 Project Structure

```
video_rag/
├── app/
│   ├── driver.py               # Main controller: handles per-video pipeline state, indexing, and coordination
│   └── main.py                 # Streamlit UI: handles layout, chat input/output, and UI triggers
│
├── assets/
│   ├── driver/                 # Driver configuration for resuming sessions post shutdown
│   ├── media/                  # Uploaded or saved video files
│   ├── frames/                 # Extracted keyframes from videos (image files)
│   └── transcriptions/         # Timestamped text segments from the audio transcript (chunked .txt files)
│
├── core/
│   ├── audio_transcriber.py    # Transcribes audio using Whisper with options for 5 different parsing models
│   ├── hit_reranker.py         # Logic to rerank search results based on context and query similarity
│   ├── image_embedder.py       # Generates dense embeddings from keyframes using any CLIP model
│   ├── keyframe_extractor.py   # Extracts keyframes from video using scene detection
│   ├── media_loader.py         # Loads and processes media assets: video, audio, transcripts, keyframes
│   ├── text_embedder.py        # Converts transcript chunks into dense embeddings using Sentence-BERT
│   └── utils.py                # Common helper functions shared across core components
│
├── retrieval/
│   ├── base_index.py           # Abstract base class for all retrieval indexes (text or image)
│   ├── bm25_index.py           # BM25-based lexical retriever over text segments (ideal for text)
│   ├── faiss_index.py          # FAISS-based dense retriever over vector embeddings (ideal for text/image)
│   ├── pgvector_index.py       # PostgreSQL pgvector-based dense retrieval (DB-based system for text/image)
│   ├── tfidf_index.py          # TF-IDF-based lexical retriever (ideal for text)
│   └── utils.py                # Shared utilities
│
├── data/
│   # Place for saving index data
│
├── evaluation/
│   ├── evaluate_retrieval.py   # Script to run quantitative evaluation on retrieval performance
│   └── gold_test_set.json      # A set of annotated gold-standard queries for testing retrieval accuracy
│
├── config.py                   # Central configuration file: paths, thresholds, model choices
└── requirements.txt            # Python dependencies for the entire system
```

---

## ✨ Key Features

### 🔍 Hybrid Multimodal Retrieval

This system uses **two independent retrieval pipelines**, then fuses their results to answer a query:

- **Audio-based retrieval (text)**: Segments the video transcript and applies lexical or dense search to retrieve spoken content related to the query.
- **Visual-based retrieval (image)**: Extracts keyframes then retrieves semantically similar text using dense vector search.

The best hits from both pipelines are reranked and returned with contextual metadata (timestamps, frame previews, segment text). Users can opt to either text retrieval, image retrieval, or both.

---

### 🧠 Modular and Pluggable Retrieval Backends

Each retrieval method is isolated under the `retrieval/` module and can be swapped or extended independently:

- **BM25Index**: A probabilistic lexical model ideal for retrieving transcript segments using exact or fuzzy keyword matches.
- **TFIDFIndex**: Classic sparse vector-based retriever using term frequency and inverse document frequency. Lightweight and interpretable.
- **FAISSIndex**: Embedding-based dense retriever using inner product. High-performance search for semantically similar captions.
- **PGVectorIndex**: Embeds vector search in PostgreSQL using the `pgvector` extension. Suitable for production settings with persistent data.
- All indexes implement a common interface with `.add()` and `.query()` methods.

---

### 🎥 Full Audio-Visual Processing Pipeline

The core pipeline includes:

- `audio_transcriber.py`: Transcribes and chunks text from video audio.
- `keyframe_extractor.py`: Selects keyframes based on scene change for summarizing visual content.
- `image_embedder.py`: Embeds frames into vector representations.
- `text_embedder.py`: Embeds transcript segments for dense search and hybrid reranking.
- `hit_reranker.py`: Combines and reranks search hits from audio and video streams to determine the best answer.

The `Driver` object in `app/driver.py` binds these stages per video and manages the session state.

---

### 💬 Streamlit-Based Interactive UI

- Clean, modular layout built with Streamlit in `main.py`.
- Bottom-fixed chat input with real-time Q&A display.
- Sidebar to select/reset videos and show metadata/configurations.

---

### 📊 Evaluation Tools

- `evaluate_retrieval.py` provides an evaluation pipeline for retrieval quality using a gold-standard test set.
- Use `gold_test_set.json` to measure accuracy and latency for different retriever combinations or query types.

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/video_rag.git
   cd video_rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
  ```bash
  streamlit run app/main.py
  ```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧾 License

MIT License. Feel free to use, modify, or redistribute.

---

Enjoy asking your videos questions they can finally answer. 🎥🧠
