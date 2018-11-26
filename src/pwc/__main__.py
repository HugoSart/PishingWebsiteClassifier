from pwc.classifier import DataProcessor
from pwc.classifier import Website
from pwc.ze_classifier import ZeClassifier


def run_test(classifier, test_dataset):

    hit = miss = 0

    for site in test_dataset:
        c = classifier.classify(site)

        if site.attributes[Website.Tag.RESULT.value] == c:
            hit += 1
        else:
            miss += 1

    print((hit / (hit + miss)) * 100, "% - ", (miss / (hit + miss)) * 100, "%")


def __main__():
    processor = DataProcessor()
    processor.build_dataset("dataset.csv")

    classifier = ZeClassifier(processor.train_dataset)
    classifier.train()

    run_test(classifier, processor.test_dataset)


if __name__ == "__main__":
    __main__()
