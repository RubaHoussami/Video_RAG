from pathlib import Path
from typing import List
import cv2
import numpy as np


class KeyframeExtractor:
    """
    Detects scene changes by histogram-difference and saves keyframes.
    """

    def __init__(self, scene_threshold: float = 0.5, bins: int = 64) -> None:
        self.scene_threshold = scene_threshold
        self.bins = bins

    def set_output_dir(self, media_name: str) -> None:
        self.output_dir = Path(f"assets/frames/{media_name}")
        if self.output_dir.exists():
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def extract(self, video_name: str, min_duration: float = 5.0) -> None:
        media_path = Path(f"assets/media/{video_name}.mp4")
        if not media_path.exists():
            raise FileNotFoundError(media_path)

        if self.set_output_dir(video_name):
            video_capture = cv2.VideoCapture(str(media_path))
            if not video_capture.isOpened():
                raise RuntimeError(f"Cannot open video {media_path}")

            fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps

            success, frame = video_capture.read()
            if not success:
                video_capture.release()
                raise RuntimeError(f"Cannot read video {media_path}")

            prev_histogram = self._compute_histogram(frame)
            frame_index = 1
            raw_boundaries = [0.0]
            raw_frames = [frame]

            while success:
                timestamp = frame_index / fps
                success, frame = video_capture.read()
                frame_index += 1
                if not success:
                    break

                histogram = self._compute_histogram(frame)
                distance = cv2.compareHist(prev_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)

                if distance > self.scene_threshold:
                    raw_boundaries.append(timestamp)
                    raw_frames.append(frame)

                prev_histogram = histogram

            video_capture.release()

            scene_boundaries, frames = self._merge_short_segments(raw_boundaries, raw_frames, min_duration)

            for i in range(len(scene_boundaries) - 1):
                start = round(scene_boundaries[i], 2)
                end = round(scene_boundaries[i + 1], 2)
                file_path = self.output_dir / f"{start}+{end}.jpg"
                cv2.imwrite(str(file_path), frames[i])

            start = round(scene_boundaries[-1], 2)
            end = round(duration, 2)
            file_path = self.output_dir / f"{start}+{end}.jpg"
            cv2.imwrite(str(file_path), frames[-1])

    def _compute_histogram(self, frame) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [self.bins, self.bins], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    def _merge_short_segments(self, boundaries: List[float], frames: List[np.ndarray], min_duration: float = 5.0) -> tuple[list[float], list[np.ndarray]]:
        filtered_boundaries = [boundaries[0]]
        filtered_frames = [frames[0]]

        for i in range(1, len(boundaries)):
            last_start = filtered_boundaries[-1]
            current = boundaries[i]
            if current - last_start >= min_duration:
                filtered_boundaries.append(current)
                filtered_frames.append(frames[i])

        return filtered_boundaries, filtered_frames
