from datasets import load_dataset

gs = load_dataset("speechcolab/gigaspeech", "xs", token='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')

print(gs)

audio_input = gs["train"][0]["audio"]
transcription = gs["train"][0]["text"]

print(audio_input)