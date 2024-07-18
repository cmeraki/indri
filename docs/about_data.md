# About data

This doc covers the kind of data and tasks we would want to collect for training Indir

## Data

1. Pure audio data - from podcasts, YouTube videos, etc. in English and Indic languages
2. Turn-taking audio conversations with tagged speakers - from movies, podcasts, talk shows, interviews, radio shows, calls
   1. Can be used to teach the model - how to have a conversation
   2. If no speaker tags are available - need diarization
3. Parallel data (Audio, Transcripts)
   1. STT can be used for clean audio data
   2. TTS can be used for clean text data - scripts, QnA
   3. Transliteration to English if transcripts are in native language
   4. Speech to speech translation if less indic language is present

## Tasks

1. STT or TTS tasks
2. Voice cloning
3. (Text, Audio) or (Audio, Text) continuations
   1. Speak this aloud.., write and tell me..
   2. The model should learn how and when to alternate between text and audio
4. Multimodal in/out
   1. Text+Audio input, Text output = Tell me which bird is this <Audio>...
   2. Text+Audio input, Audio output = Summarize this speech in the same voice as the speaker <Audio>... <Audio>

## Links for later reference

1. [Kayra](https://karya.in/) - This company helps collect data
