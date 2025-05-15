from pathlib import Path
from typing import Literal
import torch
import whisper
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim


class AudioTranscriber:
    """
    Transcribe a video / audio file using OpenAI's Whisper model.
    """

    def __init__(self, model_size: Literal['tiny', 'base', 'small', 'medium', 'large', 'turbo']) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model(model_size, device=device)
        self.model = self.model.to(device)
        self.model.eval()

    def set_output_dir(self, media_name: str, language: str) -> None:
        self.output_dir = Path(f"assets/transcriptions/{media_name}-{language}")
        if self.output_dir.exists():
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _detect_language(self, path: str) -> str:
        audio = load_audio(path)
        mel = log_mel_spectrogram(pad_or_trim(audio)).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def transcribe(self, video_name: str, min_duration: float = 5.0, beam_size: int = None, temperature: float = None, **kwargs) -> str:
        media_path = Path(f"assets/media/{video_name}.mp4")
        if not media_path.exists():
            raise FileNotFoundError(media_path)

        language = self._detect_language(str(media_path))
        if self.set_output_dir(video_name, language):
            result = self.model.transcribe(str(media_path), language=language, beam_size=beam_size, temperature=temperature, condition_on_previous_text=True, **kwargs)
            merged_result = self._merge_short_segments(result.get("segments", []), min_duration=min_duration)
            
            for segment in merged_result:
                start = round(float(segment['start']), 2)
                end = round(float(segment['end']), 2)
                text = segment['text'].strip()
                file_path = self.output_dir / f"{start}+{end}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
        return language

    def _merge_short_segments(self, segments, min_duration=5.0):
        merged = []
        i = 0
        n = len(segments)

        while i < n:
            start = segments[i]['start']
            end = segments[i]['end']
            text = segments[i]['text'].strip()

            if (end - start) >= min_duration:
                merged.append({'start': start, 'end': end, 'text': text})
                i += 1
            else:
                j = i + 1
                while j < n and (segments[j]['end'] - start) < min_duration:
                    text += ' ' + segments[j]['text'].strip()
                    end = segments[j]['end']
                    j += 1

                if j < n:
                    text += ' ' + segments[j]['text'].strip()
                    end = segments[j]['end']
                    j += 1

                merged.append({'start': start, 'end': end, 'text': text})
                i = j

        return merged

    def get_mapping(self, video_name: str, language: str) -> dict[str, str]:
        if self.set_output_dir(video_name, language):
            raise FileNotFoundError(self.output_dir)
        
        mapping = {}

        for file_path in sorted(self.output_dir.glob("*.txt")):
            index = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            mapping[index] = text

        return mapping
