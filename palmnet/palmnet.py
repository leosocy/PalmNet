import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from palmnet.dataset import load_train_test_dataset
from palmnet.layers import ArcLayer, GaborConv2D
from palmnet.losses import ArcLoss


def train(dataset_names=None, epochs=64, batch_size=32):
    train_dataset, test_dataset, label_names = load_train_test_dataset(names=dataset_names, batch_size=batch_size)
    num_classes = len(label_names)

    model = keras.Sequential(
        [
            GaborConv2D(12, 5, trainable=False),
            MaxPooling2D((2, 2)),
            Conv2D(12, 3, activation="relu"),
            MaxPooling2D((4, 4)),
            Conv2D(6, 3, activation="relu"),
            Flatten(),
            Dense(num_classes),
            ArcLayer(num_classes),
        ],
    )

    model.compile(optimizer="adam", loss=ArcLoss(), metrics=["accuracy"])
    model.build((32, 128, 128, 1))
    model.summary()

    model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset)
    model.save("./palmnet.h5")
