from pathlib import Path
import os
import re
import unicodedata
import hashlib
from yt_dlp import YoutubeDL
from core.utils import Media


class MediaLoader:
    """
    Downloads media from a given URL or uploads from user input and saves it to a local directory.
    """

    def __init__(self) -> None:
        self.base_dir = Path("assets/media")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if os.path.exists("assets/media/media_list.txt"):
            with open("assets/media/media_list.txt", "r") as f:
                self.media = {}
                for line in f:
                    title, hashed = line.strip().split(",")
                    self.media[hashed] = Media(title=title, hashed=hashed)
        else:
            self.media = {}

    def save_state(self) -> None:
        with open("assets/media/media_list.txt", "w") as f:
            for media in self.media.values():
                f.write(f"{media.title},{media.hashed}\n")

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sanitize_title(title: str) -> str:
        title = unicodedata.normalize("NFKD", title)
        title = title.encode("ascii", "ignore").decode("ascii")
        title = re.sub(r'[\\\\/:\"*?<>|,]', "_", title)
        title = re.sub(r"\\s+", "_", title)
        title = re.sub(r"_+", "_", title).strip("_")
        return title.lower()

    def download(self, url: str) -> Media:
        hashed_url = self.hash_url(url)
        target_path = self.base_dir / f"{hashed_url}.mp4"

        if not target_path.exists():
            with YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
            
            title = self.sanitize_title(title)
            media = Media(title=title, hashed=hashed_url)
            ydl_opts = {
                'outtmpl': str(target_path),
                'format': 'best'
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                self.media[hashed_url] = media
                return media
        return self.media[hashed_url]

    def upload(self, uploaded_file) -> Media:
        file_name = uploaded_file.name
        hashed_url = self.hash_url(file_name)
        target_path = self.base_dir / f"{hashed_url}.mp4"

        if not target_path.exists():
            title = self.sanitize_title(file_name)
            media = Media(title=title, hashed=hashed_url)

            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
            self.media[hashed_url] = media
            return media
        return self.media[hashed_url]
    
    def delete(self, hashed: str) -> None:
        media_path = self.base_dir / f"{hashed}.mp4"
        frames_path = Path(f"assets/frames/{hashed}")
        transcriptions_path = Path(f"assets/transcriptions/{hashed}")
        
        if media_path.exists():
            os.remove(media_path)
        if frames_path.exists():
            os.remove(frames_path)
        if transcriptions_path.exists():
            os.remove(transcriptions_path)
        if hashed in self.media:
            del self.media[hashed]

    def list_media(self) -> list[str]:
        return sorted([medium.title for medium in self.media.values()])

    def get_hash(self, title: str) -> str:
        for medium in self.media.values():
            if medium.title == title:
                return medium.hashed
        return None
