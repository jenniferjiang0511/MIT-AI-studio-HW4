'''
Jennifer Jiang
jenniferjiang@g.harvard.edu, jiang511@mit.edu
MIT AI studio HW 4
'''
import os

'''since I am uploading my code, I have also used placeholder API keys
but I have used actual API keys for generating output'''

os.environ["OPENAI_API_KEY"] = "ABC"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

"""
Unified STT + TTS + Persona (CrewAI)
- Input:
    • If user enters a local audio file path -> STT (Whisper), then persona responds in text.
    • If user enters plain text -> persona responds in text, then TTS (Kokoro) speaks the persona's reply.

Outputs (in outputs/):
    - transcript.txt / .vtt / .srt / _segments.json (if audio input)
    - reply.txt  (persona reply, always)
    - reply_speech.wav  (only if text input, i.e., persona "talks back")
"""


import sys
import json
import shutil
import tempfile
from urllib.parse import urlparse
from typing import Optional
from crewai import Crew, Process, Task, Agent
from crewai.tools import tool
# pip install -U openai-whisper kokoro soundfile numpy crewai
import whisper
import soundfile as sf
import numpy as np
from kokoro import KPipeline

from pathlib import Path
from urllib.parse import urlparse

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mkv", ".aac"}

def _script_base_dir() -> Path:
    # robust base dir (works from IDEs); falls back to CWD if __file__ is missing
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def _resolve_audio_path(user_str: str) -> Path | None:
    """Try multiple locations to resolve a local audio path."""
    if not user_str:
        return None
    s = user_str.strip().strip('"').strip("'")
    # URL? (treat as audio input handled by STT)
    parsed = urlparse(s)
    if parsed.scheme in {"http", "https"}:
        return None  # URL is handled by STT tool directly; return None to indicate "not a local file"
    # Expand ~ and env vars
    p = Path(os.path.expandvars(os.path.expanduser(s)))

    # Candidate locations to try
    base = _script_base_dir()
    candidates = [
        p,
        (base / p),
        (base / "inputs" / p.name),         # "foo.m4a" lives under ./inputs
        (base / "inputs" / p),              # "inputs/foo.m4a" when run from elsewhere
    ]

    for cand in candidates:
        if cand.is_file() and cand.suffix.lower() in AUDIO_EXTS:
            return cand.resolve()
    return None

def detect_mode(user_str: str) -> tuple[str, str | Path]:
    """
    Returns ("audio", Path) if we find a valid local audio file
            ("text", str)   otherwise (including URLs, which STT can download)
    """
    # Local file?
    local = _resolve_audio_path(user_str)
    if local is not None:
        return ("audio", local)

    # URL? Treat as audio (STT will download)
    parsed = urlparse(user_str.strip())
    if parsed.scheme in {"http", "https"}:
        return ("audio", user_str.strip())

    # Otherwise, plain text
    return ("text", user_str)

# ---------- Make sure ffmpeg is visible to Whisper ----------
def ensure_ffmpeg_on_path():
    """
    Ensures ffmpeg is discoverable by the running Python process.
    This helps when IDE PATH differs from your shell PATH.
    """
    if shutil.which("ffmpeg"):
        return
    # Try common macOS brew locations
    candidates = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
    for c in candidates:
        if os.path.exists(c):
            os.environ["PATH"] = os.path.dirname(c) + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                return
    # Optional vendored fallback
    try:
        import imageio_ffmpeg as iio
        ff = iio.get_ffmpeg_exe()
        os.environ["PATH"] = os.path.dirname(ff) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    if not shutil.which("ffmpeg"):
        sys.exit(
            "ffmpeg not found. Install it (e.g., `brew install ffmpeg`) "
            "and/or ensure its bin dir is on PATH."
        )

ensure_ffmpeg_on_path()


# ---------- Small file writer tool (to persist persona reply) ----------
@tool("write_text_file")
def write_text_file(content: str, output_dir: str = "outputs", filename: str = "reply.txt") -> str:
    """Save `content` to a text file at `output_dir/filename` and return the saved path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    return f"Saved: {path}"


# ---------- STT tool (local Whisper) ----------
def _download_to_tmp(url: str) -> str:
    import urllib.request
    suffix = os.path.splitext(urlparse(url).path)[1] or ".audio"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        with urllib.request.urlopen(url) as r:
            f.write(r.read())
    return tmp_path


def _fmt_ts(t: float, srt: bool = False) -> str:
    ms = int(round((t - int(t)) * 1000))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    sep = "," if srt else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def _write_captions(segments, vtt_path: str, srt_path: str):
    # VTT
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{seg.get('text','').strip()}\n\n")
    # SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(
                f"{i}\n{_fmt_ts(seg['start'], True)} --> {_fmt_ts(seg['end'], True)}\n{seg.get('text','').strip()}\n\n"
            )


@tool("whisper_stt")
def whisper_stt(
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    language: Optional[str] = None,          # e.g., "en"
    model_size: str = "base",                 # "tiny", "base", "small", "medium", "large"
    output_dir: str = "outputs",
    basename: str = "transcript",
    initial_prompt: Optional[str] = None,     # bias spellings/jargon
    temperature: float = 0.0
) -> str:
    """
    Transcribe an audio file with local openai-whisper and save text + captions.
    Provide exactly one of: file_path or file_url
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve input (local or URL)
    local_path = None
    if file_path and os.path.exists(file_path):
        local_path = file_path
    elif file_url:
        local_path = _download_to_tmp(file_url)
    else:
        return "Provide either a valid file_path or file_url."

    # Load Whisper model
    model = whisper.load_model(model_size)

    # Transcribe
    result = model.transcribe(
        local_path,
        language=language,
        initial_prompt=initial_prompt,
        temperature=temperature,
        condition_on_previous_text=True,
        verbose=False
    )

    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []

    # Write artifacts
    txt_out = os.path.join(output_dir, f"{basename}.txt")
    vtt_out = os.path.join(output_dir, f"{basename}.vtt")
    srt_out = os.path.join(output_dir, f"{basename}.srt")
    seg_out = os.path.join(output_dir, f"{basename}_segments.json")

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(text)

    with open(seg_out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    _write_captions(segments, vtt_out, srt_out)

    return "Saved:\n  " + "\n  ".join([txt_out, vtt_out, srt_out, seg_out])


# ---------- TTS tool (Kokoro) ----------
@tool("kokoro_tts")
def kokoro_tts(
    text: str | None = None,
    file_path: str | None = None,     # NEW
    voice: str = "af_heart",
    lang_code: str = "a",
    output_dir: str = "outputs",
    filename: str = "reply_speech.wav",
    sample_rate: int = 24000,
    preview_seconds: float = 0.0
) -> str:
    """Synthesize speech from `text` or from the contents of `file_path` (24 kHz WAV)."""
    # --- read text if file_path provided ---
    if file_path:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

    if not text or not text.strip():
        return "No input text provided to synthesize."

    os.makedirs(output_dir, exist_ok=True)
    pipe = KPipeline(lang_code=lang_code)
    chunks = [a for _, _, a in pipe(text, voice=voice)]
    if not chunks:
        return "No audio produced."

    import numpy as np, soundfile as sf
    audio = np.concatenate(chunks)
    full_path = os.path.join(output_dir, filename)
    sf.write(full_path, audio, sample_rate)

    artifacts = [full_path]
    if preview_seconds and preview_seconds > 0:
        pv = audio[: int(preview_seconds * sample_rate)]
        preview_path = os.path.join(output_dir, "reply_preview.wav")
        sf.write(preview_path, pv, sample_rate)
        artifacts.append(preview_path)

    return "Saved:\n  " + "\n  ".join(artifacts)



# ---------- Agents ----------
def create_stt_agent() -> Agent:
    return Agent(
        role="Speech-to-Text Transcriber",
        goal="Transcribe audio into accurate text with captions and segment timings.",
        backstory="Skilled at converting spoken content to clean transcripts.",
        verbose=True,
        allow_delegation=False,
        tools=[whisper_stt, write_text_file],
    )


def create_tts_agent() -> Agent:
    return Agent(
        role="Text-to-Speech Producer",
        goal="Convert persona replies into natural, intelligible speech (24 kHz WAV).",
        backstory="Audio specialist focused on clear delivery and consistent loudness.",
        verbose=True,
        allow_delegation=False,
        tools=[kokoro_tts],
    )


def create_persona_agent() -> Agent:
    return Agent(
        role="PhD Student Conversationalist",
        goal=("Reply like a friendly PhD student who studies proteins, "
              "plays tennis, and enjoys explaining science clearly."),
        backstory=("You are a PhD student at Harvard University, studying how proteins interact with each other on a molecular level."
              "You perform research and experiments every day. Outside of the lab, you like to play tennis. "
                   "You also enjoy trying new restaurants and foods. Your voice is warm, supportive, and practical."),
        verbose=True,
        allow_delegation=False,
        tools=[write_text_file, kokoro_tts],  # can save reply, and (for text inputs) request TTS
    )


# ---------- Tasks ----------
def create_audio_to_text_pipeline_tasks(stt_agent: Agent, persona_agent: Agent, audio_path: str):
    """
    For audio input:
      1) STT -> transcript.* in outputs/
      2) Persona reads transcript and writes a text reply to outputs/reply.txt
    """
    stt_task = Task(
        description=f"""Transcribe this local audio file:

{audio_path}

Write:
- outputs/transcript.txt
- outputs/transcript.vtt
- outputs/transcript.srt
- outputs/transcript_segments.json
""",
        expected_output="""Status string listing saved transcript artifacts.""",
        agent=stt_agent
    )

    persona_task = Task(
        description="""Read 'outputs/transcript.txt'. 
Write a concise, supportive reply as a PhD student who studies proteins and plays tennis.
Tone: warm, encouraging, a bit nerdy.
Save only your reply text to 'outputs/reply.txt' using the write_text_file tool.""",
        expected_output="Saved: outputs/reply.txt",
        agent=persona_agent
    )

    return [stt_task, persona_task]


def create_text_to_speech_pipeline_tasks(persona_agent: Agent, tts_agent: Agent, user_text: str):
    """
    For text input:
      1) Persona crafts a reply and saves it to outputs/reply.txt
      2) TTS speaks the persona's reply to outputs/reply_speech.wav
    """
    persona_task = Task(
        description=f"""User wrote:

{user_text}

Reply in the style of a PhD student who studies proteins and plays tennis.
Be kind, a touch playful, and specific. 
Length: 2–5 sentences.
Then save ONLY your reply text to 'outputs/reply.txt' using the write_text_file tool.""",
        expected_output="Saved: outputs/reply.txt",
        agent=persona_agent
    )

    tts_task = Task(
        description="""Load the text from 'outputs/reply.txt' and synthesize it to speech 
with Kokoro (voice 'af_heart', US English, 24 kHz). 
Save to 'outputs/reply_speech.wav'.""",
        expected_output="Saved: outputs/reply_speech.wav",
        agent=tts_agent
    )

    return [persona_task, tts_task]


# ---------- Main orchestration ----------
def main():
    print("🎛️  Unified STT + TTS + Persona (CrewAI)")
    print("=" * 60)

    user_input = input(
        "Enter EITHER:\n"
        "  • a local AUDIO FILE path (e.g., inputs/sample.m4a)\n"
        "  • plain TEXT (I will reply and speak it)\n\n> "
    ).strip()

    if not user_input:
        print("No input provided. Exiting.")
        return

    # Decide mode: audio path vs text
    is_audio_file = os.path.exists(user_input) and os.path.isfile(user_input)

    # Agents
    stt_agent = create_stt_agent()
    tts_agent = create_tts_agent()
    persona_agent = create_persona_agent()

    if is_audio_file:
        print("\n🎧 Detected local audio file → STT + persona TEXT reply")
        tasks = create_audio_to_text_pipeline_tasks(stt_agent, persona_agent, user_input)
    else:
        print("\n⌨️  Detected TEXT input → persona reply + TTS")
        tasks = create_text_to_speech_pipeline_tasks(persona_agent, tts_agent, user_input)

    crew = Crew(
        agents=[stt_agent, tts_agent, persona_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    print("\n🚀 Running crew...")
    print("=" * 60)
    try:
        result = crew.kickoff()
        print("\n✅ Done!")
        print("=" * 60)
        print("📄 Final Result:")
        print(result)

        # Summarize likely artifacts
        print("\n📦 Output artifacts (check ./outputs):")
        if is_audio_file:
            for f in ["transcript.txt", "transcript.vtt", "transcript.srt", "transcript_segments.json", "reply.txt"]:
                p = os.path.join("outputs", f)
                print(" •", p, "(exists)" if os.path.exists(p) else "(missing)")
        else:
            for f in ["reply.txt", "reply_speech.wav"]:
                p = os.path.join("outputs", f)
                print(" •", p, "(exists)" if os.path.exists(p) else "(missing)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("- Ensure ffmpeg is installed & on PATH (the script tries to patch PATH but may need a shell restart).")
        print("- For audio inputs, supported formats depend on ffmpeg (e.g., .m4a/.mp3/.wav/.flac).")
        print("- If a URL was entered by mistake, download it first and pass a local path.")


if __name__ == "__main__":
    main()
