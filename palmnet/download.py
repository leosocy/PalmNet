import argparse
import io
import logging
import os
import tarfile

import requests
import tqdm

from palmnet.dataset import SupportedDatasetName


def download_dataset(dataset: SupportedDatasetName):
    if os.path.exists(f"./{dataset.value}"):
        logging.warning(f"dataset {dataset.value} already exists, skip download")
        return

    buffer = io.BytesIO()
    response = requests.get(dataset.oss_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, desc=dataset.value) as bar:
        for data in response.iter_content(chunk_size=1024):
            buffer.write(data)
            bar.update(len(data))

    buffer.seek(0)
    file = tarfile.open(fileobj=buffer, mode="r|gz")
    file.extractall(path=f"./datasets/{dataset.value}")
