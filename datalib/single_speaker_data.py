from huggingface_hub import snapshot_download

urls = ["openslr/librispeech_asr",
        "ylacombe/expresso",
        "MushanW/GLOBE",
        "MikhailT/hifi-tts",
        "IVLLab/MultiDialog",
        ]


for url in urls:
    snapshot_download(local_dir=f'/media/apurva/HD-PCTU3/indri_data/single_speaker/{url}', repo_id=url, repo_type='dataset')


snapshot_download(local_dir='/media/apurva/HD-PCTU3/indri_data/peoples_speech/',
                  repo_id='MLCommons/peoples_speech',
                  repo_type='dataset',
                  allow_patterns='train/clean/clean*.tar')

