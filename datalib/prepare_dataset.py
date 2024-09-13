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
    
    chosen_files = ['084', '089', '088', '085', '080', '087', '082', '081', 
                    '086', '083', '068', '063', '060', '066', '067', '064', 
                    '069', '061', '062', '108', '065', '091', '101', '103', 
                    '106', '093', '094', '109', '105', '104', '100', '096', 
                    '097', '102', '107', '095', '099', '098', '090', '092', 
                    '073', '079', '072', '077', '071', '070', '561', '075', 
                    '562', '076', '074', '078', '567', '568', '566', '560', 
                    '563', '564', '565', '569', '133', '137', '551']

    chosen_files = ['EN/EN-B000' + x + '.tar' for x in chosen_files]
    print(chosen_files)
    dataset = load_dataset(
                path=dconfig['path'],
                streaming=False,
                data_files={"en": chosen_files}, split="en")
                # **dconfig)
    
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

def prepare(dsname, num):
    dataset = Dataset(repo_id=dsname)
    prep_method = get_prepare_method(dsname)
    
    for item in iter_hf_item(dsname, num, streaming=True):
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
        prepare(args.dsname,
                args.num)

    else:
        print("Pass a valid mode")
