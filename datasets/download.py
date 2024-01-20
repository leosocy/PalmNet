import argparse
import enum
import io
import logging
import os
import tarfile

import requests
import tqdm


@enum.unique
class SupportedDataset(enum.Enum):
    POLYU = "PolyU"
    TONGJI = "Tongji"

    @classmethod
    def from_str(cls, s):
        return cls[s.upper()]

    @property
    def oss_url(self):
        return f"https://blog-images-1257621236.cos.ap-shanghai.myqcloud.com/PalmNet/{self.value}.tar.gz"


def download_dataset(dataset: SupportedDataset):
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
    file.extractall(path=f"./{dataset.value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        help="download which datasets, default is all",
    )
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "all":
        for dataset in SupportedDataset:
            download_dataset(dataset)
    else:
        download_dataset(SupportedDataset.from_str(dataset))
