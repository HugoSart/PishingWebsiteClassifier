from pwc.classifier import DataProcessor
from pwc.classifier import Classifier
from pwc.classifier import Website


def run_test(classifier, test_dataset):
    for site in test_dataset:
        c = classifier.classify(site)
        if site.attributes[Website.Tag.RESULT.value] == c:
            print("Correto")
        else:
            print("Errado")


def __main__():
    processor = DataProcessor()
    processor.build_dataset("dataset.csv")

    classifier = Classifier
    classifier.train_dataset()

    run_test(classifier, processor.test_dataset)


__main__()
