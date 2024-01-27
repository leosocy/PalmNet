import argparse

import palmnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="datasets",
        metavar="d",
        type=palmnet.SupportedDatasetName,
        default=[],
        nargs="+",
        help="train which datasets, default is all",
    )
    parser.add_argument("--epochs", type=int, default=64, help="epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--predict", action="store_true", default=False, help="predict with trained palmnet model")
    parser.add_argument("--download", action="store_true", default=False, help="download dataset")
    args = parser.parse_args()

    if args.download:
        for dataset_name in args.datasets or list(palmnet.SupportedDatasetName):
            palmnet.DatasetRegistry.get_dataset(dataset_name).download()
    elif args.predict:
        palmnet.predict(dataset_names=args.datasets)
    else:
        palmnet.train(dataset_names=args.datasets, epochs=args.epochs, batch_size=args.batch_size)
