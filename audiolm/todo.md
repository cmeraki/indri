1. create val set out if librispeech dev
2. tokens for all
3. generation + sampling + inference
4. setup wandb account
5. setup eternal training
8. next datasets -> 10k hours

train : 
1M tokens => 4s
10B tokens => 40000s => 11 hours => 10k steps
val : 
10M tokens => 40s

train params :
1. seq length = 4096