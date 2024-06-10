Workbook
========

D 1. create val set out if librispeech dev
D 2. tokens for all
D 3. generation + sampling + inference 
4. cleanup training script
W 5. setup wandb account
6. setup eternal training
7. next datasets -> 10k hours

3-10/6
 0. extend tokenizer for large audio files
D 1. continuous inference : listen 1/2 length, generate rest, repeat, demo copy cat with continuation
D 2. sample of speech, continue speaking
D 3. 10B training run on jarvis, decide dataset => completed with 20B
S 4. Finalize jarvis setup, notebooks OR fabfiles
D 5. Conversation data plan , movies and tornadoes
D 6. 1T audio data + training resource plan

7/6 - 10/6
1. Redirect to indic cheap
2. FT / train bark 2nd layer and test results
3. quality metrics for natural speech

Notes
=====
Indic TTS

1. For training tts you don't need a lot of data. 
2. Target is now to create a cheap and fast Indic voice system.
3. Probable cost 20ps/min of generated audio.

Spear TTS / Suno bark with Indian data
1. Use model 1 and 3 from bark chain and train semantic to coarse model on gigaspeech
2. Test on Indian voice
3. Collect youtube Indian videos
4. AI4bharat voice data is very monotonic

Steps 
======

1. text/audio-semantic is clap on text and audio. requires parallel data to train. but embeds audio and text in same space. 
2. semantic to coarse can be done by just using audio. use audio-semantic to create training data. no labelled data required.
3. coarse to fine is unsupervised
4. prompt to semantic-coarse should have sample of user voice as a prompt. [this is partially labelled data]
5. prompt = [voice sample, semantic tokens] => [coarse tokens]
6. voice sample is usually 15-30 seconds long. Don't know whether we can do such a large network right now. 
7. speaker embeddings ??

10s audio clip + 10s voice sample => 3000 tokens

Speed up bark
Tune bark for Indian Data