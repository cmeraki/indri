from datasets import load_dataset
from datalib.mappings import dataset_info
from datalib.datalib import Dataset
from tqdm import tqdm
import math


def get_prepare_method(dsname):
    return dataset_info[dsname]['method']


def iter_hf_item(dsname, num=1, streaming=True):
    dinfo = dataset_info[dsname]
    dconfig = {k:dinfo[k] for k in ['path', 'split', 'name']}
    
    dataset = load_dataset(
                streaming=streaming,
                **dconfig)
    
    dataset = iter(dataset)

    for idx, item in tqdm(enumerate(dataset), desc='iterating over dataset...'):
        yield item    
        if idx >= num:
            break

def test_prepare(dsname, num):
    prep_method = get_prepare_method(dsname)
    for item in iter_hf_item(dsname, num):
        print('ITEM', item)
        sample = prep_method(item)
        print('sample', sample)

def prepare(hfds, split, num, dsname):
    dataset = Dataset(repo_id=dsname)
    prep_method = get_prepare_method(dsname)
    
    for item in iter_hf_item(hfds, split, num, streaming=False):
        sample = prep_method(item)
        dataset.add_sample(sample)
    
    dataset.tokenize()
    dataset.upload()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--mode', type=str, required=True, help='Chose one of 3 modes')
    parser.add_argument('--num', type=int, required=False, default=math.inf, help='num samples to process / print')
    parser.add_argument('--dsname', type=str, required=True, help='name of your dataset. will be uploaded to cmeraki/{inds}')

    args = parser.parse_args()
    
    if args.mode == 'print_hf':

        for item in iter_hf_item(args.dsname,
                                args.num):
            print(item)

    elif args.mode == 'test_mapping':
        test_prepare(args.dsname,  
                     args.num)
        
    elif args.mode == 'prepare':
        prepare(args.hfds, 
                args.split, 
                args.num, 
                args.dsname)

    else:
        print("Pass a valid mode")
