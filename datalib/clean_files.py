import os
import multiprocessing as mp
from tqdm import tqdm
import torchaudio

def process_file(file):
    try:
        torchaudio.load(file)
        return 0
    except:
        os.rename(file, file + '.bad')
        return 1

from glob import glob
files = glob('/home/apurva/.cache/indri/mls_eng_10k/audio/*.opus')
bad = 0
with mp.Pool(processes=8) as pool:
    for o in tqdm(pool.imap_unordered(process_file, files), total=len(files), desc='Loading...'):
        bad += o
        
print("bad", bad)