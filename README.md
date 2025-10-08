# MIT-AI-studio-HW4
Here is an agent with text-to-speech (TTS) and speech-to-text (STT) capabilities. You can either enter a text and get a response in audio, or input a local path to an audio file, and get a text response. Both responses will have my twin agent persona. 


How I implemented voice interaction (libraries, APIs, SDK usage):
The libraries that I've used include:
- CrewAI: which orchestrates “agents” and “tasks”, and is what I used for HW1 and HW2.
- I use three agents:
1) STT agent with a @tool("whisper_stt"). I integrated it with the OpenAI_whisper tool.
2) Persona agent (which is my digital twin, a PhD student at Harvard who studies proteins, plays tennis, and likes to try new foods).
3) TTS agent with a @tool("kokoro_tts"). I integrated it with the Kokoro tool. 

Libraries:
- openai-whisper is for STT
- Kokoro (KPipeline) is for TTS
- soundfile + numpy is for writing .wav files and handle audio arrays.
- pathlib and urlparse are used to resolve file paths & format caption timestamps.

In the first exmaple/run, I showed that my outputs folder is empty. I then open the program, and input a text message, "nice to meet you...". Then, I let the program run and I will get back a text reply (outputs/reply.txt) and a audio file (outputs/reply_speech.wav) corresponding to that text reply in the outputs folder. Here, if input is text, my twin agent will write in outputs/reply.txt its reply, and the TTS agent calls kokoro_tts using outputs/reply.txt and generates outputs/reply_speech.wav.

In the second example/run, I showed that my outputs folder is again empty (it doesn't have to be empty, but as the program will write over it, however, I'm showing it to show that I didn't put text or audio in there before the program ran). I then open a inputs folder, and input one of the previous audio response from example 1 that I saved. Then I let the program run and I will get back a text reply and an audio file corresponding to the text reply. In this case, the input is a local audio file, and the STT agent (whisper) creates transcript which the Persona agent reads. Then, the persona agent writes outputs/reply.txt. 

Insights: main() decides “audio mode” only if os.path.exists(user_input) is true. A path is treated as text, so I can get TTS instead of STT. To solve this, there is a detect_mode() function to see what the input is.
