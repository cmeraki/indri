Workbook
========
1. AudioPalm architecture is most promising, 
   because it allows using a pretrained text-text model to be finetuned for audio. 
   And still outperforms tts, asr etc. models
2. test a semantic token generation model like hubert-147 or mms-300 or hubert-base, by training a gpt2 on semantic-coarse tokens
3. if 2 works then take a text only gpt2 and extend it to audio. 
4. fast tokenization in large amounts with hubert and encodec required

