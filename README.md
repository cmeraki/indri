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
size can be one of 30m or 125m

```
python -m tts.infer --size 30m --text 'mary had a little lamb <comma> and she was white as snow'
```


