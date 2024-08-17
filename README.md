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

### Long inference
```
python -m tts.long_infer --size 125m --text 'mary had a little lamb <comma> and she was white as snow'
```
This runs the model iteratively using prior generation as a part of context

An example of a long inference: 

https://github.com/user-attachments/assets/921a0503-1422-4d87-a541-b2f7913e79c1


