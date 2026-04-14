"""
core.py — transcription and diarization engine.
All processing runs locally; no internet access required after first model download.
"""

import array
import ssl
import tempfile
import warnings
import wave
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# ── Model caches (avoid reloading between runs) ───────────────────────────────

_whisper_model = None
_whisper_model_size = None
_diarization_pipeline = None


# ── Audio conversion ──────────────────────────────────────────────────────────

def _convert_to_wav(input_path: Path, tmp_dir: Path):
    """Convert any audio format to 16 kHz mono WAV using PyAV (no system ffmpeg needed).
    Returns (wav_path, audio_array, sample_rate)."""
    import av

    wav_path = tmp_dir / (input_path.stem + "_converted.wav")
    container = av.open(str(input_path))
    audio_stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    pcm_data = array.array("h")
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            pcm_data.frombytes(bytes(resampled.planes[0]))
    for resampled in resampler.resample(None):
        pcm_data.frombytes(bytes(resampled.planes[0]))
    container.close()

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data.tobytes())

    audio = np.array(pcm_data, dtype=np.float32) / 32768.0
    return wav_path, audio, 16000


def _load_wav(wav_path: Path):
    """Load a WAV file as a numpy float32 array. Returns (audio_array, sample_rate)."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


# ── Whisper transcription ─────────────────────────────────────────────────────

def load_whisper_model(model_size: str):
    global _whisper_model, _whisper_model_size
    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model

    import whisper

    # Bypass SSL inspection (institutional networks with self-signed certs).
    # Only active during the one-time model download; restored immediately after.
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        _whisper_model = whisper.load_model(model_size)
        _whisper_model_size = model_size
    finally:
        ssl._create_default_https_context = _orig

    return _whisper_model


def _transcribe(model, audio: np.ndarray) -> dict:
    return model.transcribe(audio, language="en", word_timestamps=True, verbose=False)


# ── Speaker diarization ───────────────────────────────────────────────────────

def _get_hf_token() -> str:
    """Read HuggingFace token from local file, falling back to HF cache."""
    token_file = Path(__file__).parent / ".hf_token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    # Fall back to token cached by huggingface_hub (if login was ever run)
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            return token
    except Exception:
        pass
    return None


def save_hf_token(token: str):
    """Save HuggingFace token to local file for future runs."""
    token_file = Path(__file__).parent / ".hf_token"
    token_file.write_text(token.strip())


def load_diarization_pipeline(hf_token: str = None):
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    from pyannote.audio import Pipeline
    import torch

    token = hf_token or _get_hf_token()
    if not token:
        raise RuntimeError(
            "HuggingFace token not found. "
            "Enter your token in the app settings panel."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    _diarization_pipeline.to(device)
    return _diarization_pipeline


def _diarize(audio: np.ndarray, sr: int, num_speakers=None, hf_token: str = None) -> list:
    import torch

    pipeline = load_diarization_pipeline(hf_token)
    waveform = torch.tensor(audio).unsqueeze(0)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    result = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)

    annotation = (
        getattr(result, "exclusive_speaker_diarization", None)
        or getattr(result, "speaker_diarization", result)
    )

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return sorted(segments, key=lambda s: s["start"])


# ── MFCC-based speaker assignment ─────────────────────────────────────────────

def _compute_mfcc(chunk: np.ndarray, sr: int, n_mfcc: int = 40):
    import librosa
    if len(chunk) < 400:
        return None
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]
    w = min(9, n_frames if n_frames % 2 == 1 else max(n_frames - 1, 1))
    if w >= 3:
        delta = librosa.feature.delta(mfcc, width=w)
        return np.concatenate([np.mean(mfcc, axis=1), np.mean(delta, axis=1)])
    return np.mean(mfcc, axis=1)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _build_profiles(whisper_result: dict, diarization: list, audio: np.ndarray, sr: int) -> dict:
    from collections import defaultdict
    buckets = defaultdict(list)

    for seg in whisper_result["segments"]:
        duration = seg["end"] - seg["start"]
        if duration < 1.5:
            continue
        best_speaker, best_overlap = None, 0.0
        for d in diarization:
            ov = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if ov > best_overlap:
                best_overlap, best_speaker = ov, d["speaker"]
        if best_speaker and (best_overlap / duration) >= 0.70:
            chunk = audio[int(seg["start"] * sr): int(seg["end"] * sr)]
            vec = _compute_mfcc(chunk, sr)
            if vec is not None:
                buckets[best_speaker].append(vec)

    return {spk: np.mean(vecs, axis=0) for spk, vecs in buckets.items()}


def _assign_speakers(whisper_result: dict, diarization: list, audio: np.ndarray, sr: int) -> list:
    profiles = _build_profiles(whisper_result, diarization, audio, sr)
    output = []

    for seg in whisper_result["segments"]:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_mid = (seg_start + seg_end) / 2
        chunk = audio[int(seg_start * sr): int(seg_end * sr)]

        # Stage 1: MFCC cosine similarity
        mfcc_speaker = None
        if profiles:
            vec = _compute_mfcc(chunk, sr)
            if vec is not None:
                mfcc_speaker = max(profiles, key=lambda s: _cosine_sim(vec, profiles[s]))

        # Stage 2: time-overlap fallback
        overlap_speaker, best_overlap = None, 0.0
        for d in diarization:
            ov = min(seg_end, d["end"]) - max(seg_start, d["start"])
            if ov > best_overlap:
                best_overlap, overlap_speaker = ov, d["speaker"]
        if overlap_speaker is None:
            overlap_speaker = min(
                diarization,
                key=lambda d: abs((d["start"] + d["end"]) / 2 - seg_mid),
            )["speaker"]

        output.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": mfcc_speaker or overlap_speaker,
            "text": seg["text"].strip(),
        })
    return output


# ── Transcript formatting ─────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h > 0 else f"{m:02d}:{s:05.2f}"


def build_transcript(segments: list) -> str:
    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    lines = []
    for seg in merged:
        ts = f"[{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}]"
        speaker = seg["speaker"].replace("SPEAKER_", "Speaker ")
        lines += [f"{speaker}  {ts}", seg["text"], ""]
    return "\n".join(lines)


# ── Public pipeline entry point ───────────────────────────────────────────────

def process(
    audio_path: str,
    model_size: str = "medium",
    num_speakers: int = 0,
    on_progress=None,
    hf_token: str = None,
) -> str:
    """
    Run the full transcription pipeline on an audio file.

    Args:
        audio_path:   Path to .m4a, .wav, .mp3, etc.
        model_size:   Whisper model size ('tiny', 'base', 'small', 'medium',
                      'large-v2', 'large-v3').
        num_speakers: Expected number of speakers. 0 = auto-detect.
        on_progress:  Optional callable(message: str) for status updates.

    Returns:
        Formatted transcript as a string.
    """
    def progress(msg):
        if on_progress:
            on_progress(msg)

    audio_path = Path(audio_path)
    speakers = num_speakers if num_speakers > 0 else None
    kwargs = {"hf_token": hf_token}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        progress("Loading Whisper model...")
        model = load_whisper_model(model_size)

        if audio_path.suffix.lower() == ".wav":
            progress("Loading audio...")
            audio, sr = _load_wav(audio_path)
        else:
            progress("Converting audio...")
            _, audio, sr = _convert_to_wav(audio_path, tmp_dir)

        progress("Transcribing speech...")
        whisper_result = _transcribe(model, audio)

        progress("Identifying speakers...")
        diarization = _diarize(audio, sr, speakers, hf_token=kwargs.get("hf_token"))

        progress("Matching speakers to transcript...")
        segments = _assign_speakers(whisper_result, diarization, audio, sr)

    progress("Done.")
    return build_transcript(segments)
