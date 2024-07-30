Workbook
========
1. AudioPalm architecture is most promising, 
   because it allows using a pretrained text-text model to be finetuned for audio. 
   And still outperforms tts, asr etc. models
2. test a semantic token generation model like hubert-147 or mms-300 or hubert-base, by training a gpt2 on semantic-coarse tokens
3. if 2 works then take a text only gpt2 and extend it to audio. 
4. fast tokenization in large amounts with hubert and encodec required

Jul 22
=======
1. get pretrained gpt2 124m for 400b, perform surgery to add tokens, finetune on instruct to test fting
2. Tokenize all audio, text and images
3. continue pretraining on 20B text+images+audio => 60B tokens
4. FT on ASR, TTS, Image generation, image captioning tasks

Build a UI to try speech to text and text to speech. 
Try longer text and speech : streaming
400b finetuned model doing audio in and audio out
