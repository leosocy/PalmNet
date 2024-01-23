import abc
import enum
import os.path
import re
from functools import cached_property
from typing import List

import cv2 as cv
import keras
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit


@enum.unique
class SupportedDatasetName(enum.Enum):
    POLYU = "PolyU"
    TONGJI = "Tongji"

    @classmethod
    def from_str(cls, s):
        return cls[s.upper()]

    @property
    def oss_url(self):
        return f"https://blog-images-1257621236.cos.ap-shanghai.myqcloud.com/PalmNet/{self.value}.tar.gz"


class Palmprint:
    def __init__(self, label: str, image_path: str, image_size: cv.typing.Size = (128, 128)):
        self._label = label
        self._image_path = image_path
        self._image_size = image_size

    @property
    def label(self) -> str:
        return self._label

    @property
    def image_path(self):
        return self._image_path

    @cached_property
    def image(self) -> cv.typing.MatLike:
        img = cv.imread(self._image_path, cv.IMREAD_GRAYSCALE)
        return cv.resize(img, self._image_size)


class Dataset:
    _COS_BASE_PATH = "https://blog-images-1257621236.cos.ap-shanghai.myqcloud.com/PalmNet"
    _IMAGE_NAME_REGEX = re.compile(r".*\.(png|jpg|bmp)")

    @abc.abstractmethod
    def download_url(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_palmprints(self) -> List[Palmprint]:
        raise NotImplementedError

    @staticmethod
    def listdir(path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    @staticmethod
    def listfile(path):
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    @staticmethod
    def is_image(name):
        return re.match(Dataset._IMAGE_NAME_REGEX, name) is not None


class PolyUDataset(Dataset):
    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def download_url(self) -> str:
        return f"{self._COS_BASE_PATH}/PolyU.tar.gz"

    def load_palmprints(self) -> List[Palmprint]:
        palmprints = []
        for spectral_name in self.listdir(self._base_dir):
            spectral_dir = os.path.join(self._base_dir, spectral_name)
            for identity_name in self.listdir(spectral_dir):
                identity_dir = os.path.join(spectral_dir, identity_name)
                identity = identity_name
                for image_name in self.listfile(identity_dir):
                    if not self.is_image(image_name):
                        continue
                    palmprints.append(Palmprint(f"PolyU-{identity}", os.path.join(identity_dir, image_name)))
        return palmprints


class TongjiDataset(Dataset):
    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def download_url(self) -> str:
        return f"{self._COS_BASE_PATH}/Tongji.tar.gz"

    def load_palmprints(self) -> List[Palmprint]:
        palmprints = []
        for session_name in self.listdir(self._base_dir):
            session_dir = os.path.join(self._base_dir, session_name)
            for image_name in self.listfile(session_dir):
                if not self.is_image(image_name):
                    continue
                identity = str(int((int(image_name[:-4]) - 1) / 10))
                palmprints.append(Palmprint(f"Tongji-{identity}", os.path.join(session_dir, image_name)))
        return palmprints


def load_palmprints(dataset_names: List[SupportedDatasetName] = None) -> List[Palmprint]:
    dataset_names = dataset_names or list(SupportedDatasetName)
    palmprints = []
    for dataset_name in dataset_names:
        if dataset_name == SupportedDatasetName.POLYU:
            dataset = PolyUDataset("./datasets/PolyU")
            palmprints.extend(dataset.load_palmprints())
        elif dataset_name == SupportedDatasetName.TONGJI:
            dataset = TongjiDataset("./datasets/Tongji")
            palmprints.extend(dataset.load_palmprints())
        else:
            raise ValueError(f"Unsupported dataset name {dataset_name}")
    return palmprints


def load_palmprint_dataset(names=None):
    images = []
    labels = []
    for palmprint in load_palmprints(names):
        images.append(palmprint.image)
        labels.append(palmprint.label)

    indices = np.random.permutation(len(images))
    return np.array(images)[indices], np.array(labels)[indices]


def load_train_test_dataset(names=None, test_size=0.3, batch_size=32, for_predict=False):
    x, y = load_palmprint_dataset(names=names)
    label_names = np.unique(y)
    name_lookup = keras.layers.StringLookup(
        vocabulary=list(label_names),
        num_oov_indices=0,
        mask_token=None,
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train = np.expand_dims(x_train, axis=-1).astype("float32")
        x_test = np.expand_dims(x_test, axis=-1).astype("float32")
        x_train /= 255.0
        x_test /= 255.0

        if for_predict:
            return x_train, y_train, x_test, y_test

        y_train = name_lookup(y_train)
        y_test = name_lookup(y_test)
        y_train = to_categorical(y_train, len(label_names))
        y_test = to_categorical(y_test, len(label_names))

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        return (
            train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size),
            test_dataset.batch(batch_size),
            label_names,
        )
