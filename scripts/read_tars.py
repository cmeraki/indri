from tqdm import tqdm
import tarfile
import torchaudio


def iterate_tar(tar_path):
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                file_content = tar.extractfile(member)
                yield file_content


def test_torchaudio_iter(tar_path):
    for filename in tqdm(iterate_tar(tar_path)):
        x, sr = torchaudio.load(filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--tar', required=True)

    args = parser.parse_args()

    test_torchaudio_iter(args.tar)


