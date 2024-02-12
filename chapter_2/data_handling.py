from pathlib import Path
import pandas as pd
import tarfile # to open the .tgz file
import urllib.request
import numpy as np
from zlib import crc32

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def separate_train_test_numpy(data, ratio = 0.2):
    np.random.seed(27)
    data = data.reset_index(drop = True)
    shuffled_indices = np.random.permutation(data.index)
    test_size = int(len(data)*ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    test_set = data.loc[test_indices]
    train_set = data.loc[train_indices]

    return train_set, test_set


def is_id_in_test_set(id, ratio):
    return crc32(np.int64(id)) < ratio*2**32

def separate_train_test_crc32(data, ratio = 0.2):
    data = data.reset_index()
    ids = data['index']
    in_test_index = ids.apply(lambda x: is_id_in_test_set(x, ratio))
    test_set = data.loc[in_test_index]
    train_set = data.loc[~in_test_index]

    return train_set, test_set


if __name__ == '__main__':
    housing = load_housing_data()