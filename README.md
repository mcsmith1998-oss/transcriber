# Interview Transcriber

Automatically transcribes audio interviews and labels who is speaking. Runs entirely on your own computer — audio files are never uploaded or shared.

---

## Requirements

- A Mac or Windows PC
- [Python 3.9 or later](https://www.python.org/downloads/) — when installing on Windows, tick **"Add Python to PATH"**
- A free [HuggingFace](https://huggingface.co) account (needed once to download the speaker identification model)

---

## Installation (do this once)

### Step 1 — Download the app

Click the green **Code** button on this page and select **Download ZIP**. Unzip it somewhere on your computer.

### Step 2 — Run the setup script

**On Mac:**
1. Open **Terminal** (press `Cmd + Space`, type `Terminal`, press Enter)
2. Type `cd ` (with a space), then drag the unzipped folder into the Terminal window and press Enter
3. Run:
```
bash setup.sh
```

**On Windows:**
1. Open the unzipped folder
2. Double-click **setup.bat**

This installs all the required software. It only needs to be done once.

### Step 3 — Get a HuggingFace token

The app uses an AI model for speaker identification. You need to authorise access to it once:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at these two pages (click **Agree and access repository**):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), click **New token**, give it any name, set access to **Read**, and click **Create**
4. Copy the token (it starts with `hf_`)

You'll paste this into the app in the next step.

---

## Running the app

**On Mac:** Double-click **Transcriber.command**

> If you see a security warning, go to System Settings → Privacy & Security → scroll down and click **Open Anyway**

**On Windows:** Double-click **run.bat**

A browser window will open automatically. The app runs locally on your machine — nothing is sent to the internet.

### First run only — enter your token

At the bottom of the app, open the **Settings — HuggingFace Token** panel, paste your `hf_` token, and click **Save Token**. You won't need to do this again.

The first time you transcribe, the app will download the AI models (~2 GB total). This takes a few minutes and requires an internet connection. Every run after that works fully offline.

---

## How to use

1. Click **Audio File** and select your `.m4a` or `.wav` interview recording
2. Set **Number of Speakers** (how many people are in the recording — leave at 0 if unsure)
3. Leave **Transcription Model** set to `medium` unless you need higher accuracy (`large-v3`) or faster speed (`small`)
4. Click **Transcribe**
5. The transcript will appear on the right with speaker labels and timestamps
6. Click **Download Transcript** to save it as a text file

---

## Transcription model guide

| Model | Speed | Accuracy | Best for |
|-------|-------|----------|----------|
| `tiny` / `base` | Very fast | Lower | Quick drafts |
| `small` | Fast | Good | Clear audio |
| `medium` | Moderate | Very good | Most interviews |
| `large-v3` | Slow | Best | Accented speech, noisy audio |

---

## Privacy

- The app runs on `127.0.0.1` (your computer only) — it cannot be accessed from any other device
- Audio files are read directly from your hard drive and never copied elsewhere
- Gradio analytics are disabled
- The only time an internet connection is used is during the one-time model download

---

## Troubleshooting

**"Permission denied" when running setup.sh**
Run `bash setup.sh` instead of `./setup.sh`

**App opens but speaker identification fails**
Check that you accepted the terms for both HuggingFace model pages (Step 3 above) and that your token is saved in the Settings panel.

**Transcript has incorrect speaker labels**
Set the **Number of Speakers** slider to the exact number of people in the recording rather than leaving it at 0.

**"Launch App.command" won't open on Mac**
Right-click the file, select **Open**, then click **Open** in the security prompt.
