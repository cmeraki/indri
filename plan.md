### Part 1 [2 months]
Verify end to end pipeline for training a small e2e multimodal network that takes and returns audio, images (maybe video) and text

Questions :

    1. How to tokenize audio, images (and video) ? 
    2. Different information density of audio and images can cause unstable training. 
    3. How to use a pretrained network and add modalities to it ? Is it possible to achieve reasonable accuracy with this ? 
    4. During generation how are images generated ? They are many tokens. How is audio generated ? It is also many token. How do multiple modalities interleave ? 

Steps

    1. create 10B token audio dataset
    2. Learn how to tokenize audio with encodec (train encodec from scratch to understand better)
    3. train gpt2 with just audio pretraining
    4. train gpt2 with 10B text and 10B audio
    5. take existing gpt2, extend vocab and finetune with 10B audio
    6. create 10B image token dataset
    7. repeat experiments for images

### Part 2 [2 months]

Train a larger network with all modalities

Questions :

    1. What data is needed to create good quality intermodal learning ? 
    2. Interleaved text, images, videos and audio. 
    3. What are large sources of these datasets ? 
            
Steps:
    
    1. Find how to use a pretrained large network from above experiments
    2. Finetune llama 3 small to make it e2e multimodal

### Part 3 [2 months]

Finetune for ability to talk as a friend 
