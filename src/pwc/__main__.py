from pwc.classifier import DataProcessor
from pwc.classifier import Website


def __main__():
    processor = DataProcessor()
    processor.build_dataset("dataset.csv")


__main__()
