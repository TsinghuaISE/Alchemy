# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request
from huggingface_hub import hf_hub_download

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

def _ensure_m4_triplet(root_dir="./dataset/m4", repo_id=HUGGINGFACE_REPO):
    # 使用 normpath 而不是 abspath，避免路径被错误解析为 /app/data/set/m4
    # normpath 只规范化路径，不会改变相对/绝对性质
    root_dir = os.path.normpath(root_dir)
    
    # 定义需要的文件
    files = {
        "M4-info.csv":  "m4/M4-info.csv",
        "training.npz": "m4/training.npz",
        "test.npz":     "m4/test.npz",
    }
    
    # 先检查所有文件是否都存在（避免在只读文件系统上创建目录）
    all_exist = all(os.path.exists(os.path.join(root_dir, name)) for name in files.keys())
    
    # 如果所有文件都存在，直接返回，不需要创建目录或下载
    if all_exist:
        return
    
    # 调试信息：打印缺失的文件
    print(f"[m4] Checking M4 dataset files in: {root_dir}", flush=True)
    print(f"[m4] Current working directory: {os.getcwd()}", flush=True)
    for name in files.keys():
        full_path = os.path.join(root_dir, name)
        exists = os.path.exists(full_path)
        print(f"[m4]   {name}: {'EXISTS' if exists else 'MISSING'} ({full_path})", flush=True)
    
    # 只有在文件不存在时才尝试创建目录和下载
    # 但首先检查目录是否已存在，避免在只读文件系统上创建
    if not os.path.exists(root_dir):
        try:
            os.makedirs(root_dir, exist_ok=True)
        except OSError as e:
            print(f"[m4] ERROR: Cannot create directory {root_dir}: {e}", flush=True)
            print(f"[m4] This usually means the filesystem is read-only or you don't have permissions.", flush=True)
            print(f"[m4] Please ensure M4 dataset files exist in {root_dir} before running.", flush=True)
            raise
    
    # 下载缺失的文件
    for name, remote in files.items():
        dst = os.path.join(root_dir, name)
        if not os.path.exists(dst):
            print(f"[m4] Downloading {name} from HuggingFace...", flush=True)
            try:
                # 使用 root_dir 的父目录作为 local_dir，避免路径问题
                parent_dir = os.path.dirname(root_dir)
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote,
                    repo_type="dataset",
                    local_dir=parent_dir,
                    local_dir_use_symlinks=False
                )
                print(f"[m4] Downloaded {name} to {path}", flush=True)
            except Exception as e:
                print(f"[m4] ERROR: Failed to download {name}: {e}", flush=True)
                raise


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        _ensure_m4_triplet(dataset_file, repo_id=HUGGINGFACE_REPO)
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz')
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        m4_info = pd.read_csv(info_file)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             train_cache_file if training else test_cache_file,
                             allow_pickle=True))


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    # return pd.read_csv(INFO_FILE_PATH)
