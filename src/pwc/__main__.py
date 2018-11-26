from pwc.classifier import DataProcessor


def __main__():
    processor = DataProcessor()
    processor.read_csv("dataset.csv")
    processor.split()


__main__()
