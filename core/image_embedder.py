from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
import open_clip


class ImageEmbedder:
    """
    Encodes images with a local CLIP model from OpenCLIP.
    """

    AVAILABLE_MODELS = open_clip.list_pretrained()

    def __init__(self, model_id: str = 'ViT-B-32', platform: str = 'openai', batch_size: int = 16) -> None:
        if (model_id, platform) not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model ({model_id}, {platform}) is not available in OpenCLIP.\n")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, pretrained=platform, device=device)
        self.model = self.model.to(device)
        self.device = device
        self.model.eval()
        self.batch_size = batch_size

    def embed(self, video_name: str) -> Tuple[np.ndarray, List[str]]:
        images_path = Path(f"assets/frames/{video_name}")
        if not images_path.exists():
            raise FileNotFoundError(f"Image directory does not exist: {images_path}")

        tensors = []
        image_ids = []

        for path in sorted(images_path.glob("*.jpg")):
            img = Image.open(path).convert("RGB")
            tensors.append(self.preprocess(img))
            image_ids.append(path.stem)

        vectors = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy().astype(np.float32))

        embeddings = np.vstack(vectors)
        return embeddings, image_ids

    def encode_text(self, texts: List[str]) -> np.ndarray:
        text_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = open_clip.tokenize(texts[i : i + self.batch_size]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            text_features.append(feats.cpu().numpy().astype(np.float32))

        return np.vstack(text_features)
