from datasets import load_dataset
from datalib.mappings import dataset_info
from datalib.datalib import Dataset
from tqdm import tqdm
import math
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader


def get_prepare_method(dsname):
    return dataset_info[dsname]['method']

def convert_to_list_of_dicts(data):
    """Converts a dictionary of lists to a list of dictionaries."""

    result = []
    keys = list(data.keys())

    for i in range(len(data[keys[0]])):
        row = {}
        for key in keys:
            e = data[key][i]
            if isinstance(e, dict):
                for k in e:
                    e[k] = e[k][0]
            else:
                e = e[0]
            row[key] = e
        result.append(row)

    return result

def iter_hf_item(dsname, num=1, streaming=True):
    dinfo = dataset_info[dsname]
    dconfig = {k:dinfo[k] for k in ['path', 'split', 'name']}
    
    dataset = load_dataset(
                streaming=streaming,
                **dconfig)
    
    dataset = dataset.batch(batch_size=4)
    dataloader = DataLoader(dataset, collate_fn=None, num_workers=4)
    
    progress_bar = tqdm(total=None, desc="processing dataset..", ncols=80)
    for batch in dataloader:
        flat_elems = convert_to_list_of_dicts(batch)
        progress_bar.update(len(flat_elems))
        for elem in flat_elems:
            yield elem
        if progress_bar.n > num:
            progress_bar.close()
            break
    
    progress_bar.close()
        
        

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
