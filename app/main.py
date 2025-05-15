from __future__ import annotations

import sys, types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_dummy = types.ModuleType("torch.classes"); _dummy.__path__ = []
sys.modules.setdefault("torch.classes", _dummy)

from driver import Driver

DEFAULT_CONFIG: Dict[str, Any] = {
    "processing_method": "both",
    "whisper_model": {"model_size": "base", "min_duration": 5.0, "beam_size": 5, "temperature": 0.0},
    "text_index": {"type": "bm25"},
    "transformer_model": {"model_key": "e5", "batch_size": 16},
    "keyframe_extractor": {"min_duration": 5.0, "scene_threshold": 0.5, "bins": 64},
    "image_model": {"type": "ViT-B-32", "platform": "openai", "batch_size": 16},
    "image_index": {"type": "faiss"},
    "k": 5,
    "min_score": 0.5,
    "text_weight": 0.5,
    "epsilon": 0.1,
}

VIDEO_EXTS = {"mp4"}

def main() -> None:
    if "rag_config" not in st.session_state:
        st.session_state["rag_config"] = DEFAULT_CONFIG.copy()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.set_page_config(page_title="🎥 Video RAG QA", layout="wide")

    driver = Driver()

    # ───────────────── sidebar ─────────────────
    with st.sidebar:
        header_col, cfg_col = st.columns([5, 2], gap="small")
        header_col.header("📁 Video Control Panel")

        with cfg_col:
            with st.popover("⚙️ Configure", use_container_width=True):
                cfg = st.session_state["rag_config"]

                st.markdown("### General")
                cfg["processing_method"] = st.radio(
                    "Modalities",
                    ["audio", "video", "both"],
                    index=["audio", "video", "both"].index(cfg["processing_method"]),
                    horizontal=True,
                )
                cfg["k"] = st.number_input("Top-k to fetch", 1, 20, value=int(cfg["k"]), step=1)
                cfg["min_score"] = st.slider("Min acceptable score", 0.0, 1.0, value=float(cfg["min_score"]), step=0.05)
                cfg["epsilon"] = st.slider("Rerank ε", 0.0, 1.0, value=float(cfg["epsilon"]), step=0.01)

                # Decide which sections to render based on selection
                show_audio = cfg["processing_method"] in {"audio", "both"}
                show_video = cfg["processing_method"] in {"video", "both"}

                # ───────────────────────────────────────── Audio settings ──────
                if show_audio and show_video:
                    cfg["text_weight"] = st.slider("Text weight relative to image weight", 0.0, 1.0, value=float(cfg["text_weight"]), step=0.05)

                if show_audio:
                    st.markdown("### Whisper Audio Transcription")
                    wm = cfg["whisper_model"]
                    wm["model_size"] = st.selectbox("Model size", ["tiny", "base", "small", "medium", "large", "turbo"], index=["tiny", "base", "small", "medium", "large", "turbo"].index(wm["model_size"]), key="wm_size")
                    wm["min_duration"] = st.number_input("Min segment duration", value=float(wm["min_duration"]), step=0.5, key="wm_min_dur")
                    wm["beam_size"] = st.number_input("Beam size", value=int(wm["beam_size"]), step=1, key="wm_beam")
                    wm["temperature"] = st.number_input("Temperature", value=float(wm["temperature"]), step=0.1, key="wm_temp")

                    st.markdown("### Text Retrieval")
                    cfg["text_index"]["type"] = st.selectbox("Index type", ["bm25", "tfidf", "faiss", "ivfflat", "hnsw", "flat"], index=["bm25", "tfidf", "faiss", "ivfflat", "hnsw", "flat"].index(cfg["text_index"]["type"]), key="txt_idx")
                    tm = cfg["transformer_model"]
                    tm["model_key"] = st.selectbox("Embedder model key", ["e5", "sfr", "labse", "multi", "jina"], index=["e5", "sfr", "labse", "multi", "jina"].index(tm["model_key"]), key="tm_key")
                    tm["batch_size"] = st.number_input("Text embed batch", value=int(tm["batch_size"]), step=1, key="tm_bs")

                # ───────────────────────────────────────── Video settings ──────
                if show_video:
                    st.markdown("### Keyframe Extraction")
                    kf = cfg["keyframe_extractor"]
                    kf["min_duration"] = st.number_input("KF min duration", value=float(kf["min_duration"]), step=0.5, key="kf_min_dur")
                    kf["scene_threshold"] = st.slider("Scene threshold", 0.0, 1.0, value=float(kf["scene_threshold"]), key="kf_thr")
                    kf["bins"] = st.number_input("Histogram bins", value=int(kf["bins"]), step=1, key="kf_bins")

                    st.markdown("### Image Retrieval")
                    im = cfg["image_model"]
                    im["type"] = st.text_input("CLIP/ViT model", value=im["type"], key="im_type")
                    im["platform"] = st.text_input("Platform", value=im["platform"], key="im_platform")
                    im["batch_size"] = st.number_input("Image embed batch", value=int(im["batch_size"]), step=1, key="im_bs")
                    cfg["image_index"]["type"] = st.selectbox("Image index", ["faiss", "ivfflat", "hnsw", "flat"], index=["faiss", "ivfflat", "hnsw", "flat"].index(cfg["image_index"]["type"]), key="im_idx")

                st.caption("Close the popup or click outside to apply.")

        st.markdown("---")

        mode = st.radio("📥 Input", ["Upload", "URL"], horizontal=True)
        vid_hash = None

        if mode == "Upload":
            file = st.file_uploader("📹 Video", type=list(VIDEO_EXTS))
            if file is not None:
                with st.spinner("Processing …"):
                    result = driver.process_new_media(uploaded_file=file, configuration=st.session_state["rag_config"])
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.toast(result["message"], icon="✅")
                    vid_hash = result["hash"]
        else:
            url = st.text_input("🔗 Video URL")
            if url:
                with st.spinner("Downloading …"):
                    result = driver.process_new_media(url=url, configuration=st.session_state["rag_config"])
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.toast(result["message"], icon="✅")
                    vid_hash = result["hash"]

        existing = st.selectbox("🌱 Existing", driver.list_media())
        selected_hash = driver.media_loader.get_hash(existing) if existing else None

    # ───────────────── main area ─────────────────
    active = vid_hash or selected_hash
    if not active:
        st.info("Upload or select a video to start.")
        return

    vid_path = Path(f"assets/media/{active}.mp4")

    st.header("🧠 Chat with the video")
    st.markdown("---")
    st.subheader("💬 Ask a question")
    for msg in st.session_state["messages"]: 
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Type here…")

    if query:
        with st.spinner("Searching …"):
            ans = driver.query(video_hash=active, query=query)
        if "error" in ans:
            st.error(ans["error"])
        elif ans.get("message") == "No results found.":
            st.warning("❌ Nothing relevant.")
        else:
            s = ans["start"]
            h, rem = divmod(int(s), 3600); m, s = divmod(rem, 60)
            reply_txt = f"🎯 {h:02d}:{m:02d}:{s:02d} (score {ans['score']:.2f})"
            st.session_state["messages"].append({"role": "assistant", "content": reply_txt})
            st.success(reply_txt)
            st.video(str(vid_path), start_time=int(ans["start"]))

if __name__ == "__main__":
    main()
