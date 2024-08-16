TTS
===

Trains text-semantic and semantic-acoustic models to be used for training omni

```
pip install -r requirements.txt
```

for preparing tokens refer to https://github.com/cmeraki/audiotoken/

### Training 
```
python -m tts.train
```

1. Download already prepared token data from huggingface. uses 100GB of disk space. Downloads 50GB and untar takes 1 hour.  

2. Train a model using downloaded tokens.


### Inference
Downloads pretrained models and runs inference on them.

```
python -m tts.infer --size 125m --text 'mary had a little lamb <comma> and she was white as snow'
```

### Samples
Since we don't have speaker profiles, every generation without past context, produces a different voice profiles. 

Here are a few samples generated from 125m: 

https://github.com/user-attachments/assets/0cd4684c-6082-48be-976a-81c194dddcd8

https://github.com/user-attachments/assets/c28f8715-1023-400c-82be-c4b5610dc1b6

https://github.com/user-attachments/assets/9bec98f6-a8f0-4eb2-9808-a398f680f80d

https://github.com/user-attachments/assets/1d3394d0-ba37-462f-8e35-30bb2f3db917





