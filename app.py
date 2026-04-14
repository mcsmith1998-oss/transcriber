"""
app.py — browser-based UI for the Interview Transcriber.
Run: python app.py   (opens automatically in your browser)
"""

import tempfile
from pathlib import Path

import gradio as gr

import core


STEPS = [
    "Loading Whisper model...",
    "Loading audio...",
    "Converting audio...",
    "Transcribing speech...",
    "Identifying speakers...",
    "Matching speakers to transcript...",
    "Done.",
]
STEP_FRAC = {msg: i / (len(STEPS) - 1) for i, msg in enumerate(STEPS)}


def save_token(token):
    token = (token or "").strip()
    if not token:
        return gr.update(), "Please enter a token first."
    core.save_hf_token(token)
    # Reset cached pipeline so next run picks up the new token
    core._diarization_pipeline = None
    return gr.update(value=token), "Token saved."


def transcribe(audio_file, model_size, num_speakers, hf_token, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")

    token = (hf_token or "").strip() or None

    progress(0, desc="Starting...")

    def on_progress(msg):
        frac = STEP_FRAC.get(msg, 0)
        progress(frac, desc=msg)

    try:
        transcript = core.process(
            audio_path=audio_file,
            model_size=model_size,
            num_speakers=int(num_speakers),
            on_progress=on_progress,
            hf_token=token,
        )
    except RuntimeError as e:
        if "token" in str(e).lower():
            raise gr.Error(
                "HuggingFace token missing or invalid. "
                "Enter your token in the Settings panel below and click Save."
            )
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Transcription failed: {e}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_transcript.txt", delete=False, encoding="utf-8"
    )
    tmp.write(transcript)
    tmp.close()

    return transcript, tmp.name


# ── UI layout ─────────────────────────────────────────────────────────────────

css = """
.title { text-align: center; }
.privacy-note { text-align: center; color: #555; font-size: 0.9em; margin-top: -8px; }
.footer-note { color: #888; font-size: 0.82em; margin-top: 4px; }
"""

with gr.Blocks(title="Interview Transcriber", theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("# Interview Transcriber", elem_classes="title")
    gr.Markdown(
        "All processing runs **entirely on this machine**. "
        "Audio files are read from local storage only — "
        "nothing is uploaded, transmitted, or shared. "
        "The app is bound to `127.0.0.1` (localhost) with analytics disabled.",
        elem_classes="privacy-note",
    )

    gr.Markdown("---")

    with gr.Row():
        # ── Left column: controls ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.File(
                label="Audio File",
                file_types=[".m4a", ".wav", ".mp3", ".mp4", ".ogg", ".flac"],
            )

            num_speakers = gr.Slider(
                minimum=0,
                maximum=8,
                value=2,
                step=1,
                label="Number of Speakers",
                info="Set to 0 to detect automatically",
            )

            model_size = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                value="medium",
                label="Transcription Model",
                info="Larger models are more accurate but slower",
            )

            transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

            gr.Markdown(
                "**Model guide:**  \n"
                "`medium` — recommended (1.5 GB, ~real-time)  \n"
                "`large-v3` — best accuracy (3 GB, slower)  \n"
                "`small` — fast, less accurate (500 MB)",
                elem_classes="footer-note",
            )

        # ── Right column: output ──────────────────────────────────────────────
        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=28,
                placeholder="Transcript will appear here once processing is complete...",
            )
            download_btn = gr.File(label="Download Transcript (.txt)")

    # ── Settings accordion ────────────────────────────────────────────────────
    with gr.Accordion("Settings — HuggingFace Token", open=not bool(core._get_hf_token())):
        gr.Markdown(
            "A free HuggingFace token is required for speaker identification.  \n"
            "1. Create a free account at [huggingface.co](https://huggingface.co)  \n"
            "2. Accept the model terms at "
            "[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) "
            "and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)  \n"
            "3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
            "(Read access is enough)  \n"
            "4. Paste it below and click **Save Token**"
        )
        with gr.Row():
            token_input = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_...",
                type="password",
                value=core._get_hf_token() or "",
                scale=4,
            )
            save_token_btn = gr.Button("Save Token", scale=1)
        token_status = gr.Markdown("")

        save_token_btn.click(
            fn=save_token,
            inputs=[token_input],
            outputs=[token_input, token_status],
        )

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_size, num_speakers, token_input],
        outputs=[transcript_box, download_btn],
    )

    gr.Markdown(
        "<div class='footer-note' style='text-align:center; margin-top:12px;'>"
        "First run downloads AI models (~2 GB total) — this only happens once. "
        "All subsequent runs are fully offline."
        "</div>"
    )


if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
    demo.launch(
        inbrowser=True,
        share=False,              # no public tunnel — localhost only
        server_name="127.0.0.1", # bind to loopback only, not LAN
    )
