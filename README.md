# MIT-AI-studio-HW4
Here is an agent with text-to-speech (TTS) and speech-to-text (STT) capabilities. You can either enter a text or input a local path to an audio file, and you will get a response in both text and audio, with my twin agent persona. 


How I implemented voice interaction (libraries, APIs, SDK usage):
The libraries that I've used include:
- CrewAI: which orchestrates “agents” and “tasks”, and is what I used for HW1 and HW2.
- I use three agents:
1) STT agent with a @tool("whisper_stt"). I integrated it with the OpenAI_whisper tool.
2) Persona agent (which is my digital twin, a PhD student at Harvard who studies proteins, plays tennis, and likes to try new foods).
3) TTS agent with a @tool("kokoro_tts"). I integrated it with the Kokoro tool. 
**- openai-whisper (local): whisper.load_model("base") then model.transcribe(...). No cloud calls needed. Depends on ffmpeg.**
**- Kokoro (open weights TTS): from kokoro import KPipeline; KPipeline(lang_code="a")(text, voice="af_heart") → concatenate audio chunks → write 24 kHz WAV with soundfile.**
**soundfile + numpy: Write .wav files and handle audio arrays.
Utilities: pathlib, urlparse, small helpers to resolve file paths & format caption timestamps.**


An explanation of one example run (the same one shown in your video), including what the input and output were and what insights you observed.
In the first exmaple/run, I showed that my outputs folder is empty. I then open the program, and input a text message, "nice to meet you...". Then, I let the program run and I will get back a text reply (outputs/reply.txt) and a audio file (outputs/reply_speech.wav) corresponding to that text reply in the outputs folder. Here, if input is text, my twin agent will write in outputs/reply.txt its reply, and the TTS agent calls kokoro_tts using outputs/reply.txt and generates outputs/reply_speech.wav.

In the second example/run, I showed that my outputs folder is again empty. I then open a inputs folder, and input the previous audio response from example 1. Then I let the program run and I will get back a text reply and a audio file corresponding to the text reply. In this case, the input is a local audio file, and the STT agent (whisper) creates transcript which the Persona agent reads. Then, the persona agent writes outputs/reply.txt and passes to the TTS agent to also turn it into an audio. 
