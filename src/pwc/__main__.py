from pwc.classifier import DataProcessor
from pwc.classifier import Website
from pwc.neural_classifier import NeuralClassifier


def run_test(classifier, test_dataset):

    hit_safe = miss_safe = hit_phishing = miss_phishing = 0

    for site in test_dataset:
        c = classifier.classify(site)
        if site.attributes[Website.Tag.RESULT.value] == c:
            if c == -1:
                hit_phishing += 1
            else:
                hit_safe += 1
        else:
            if c == -1:
                miss_phishing += 1
            else:
                miss_safe += 1

    print("+----------+----------+----------+")
    print("|          | SAFE     | PHISHING |")
    print("+----------+----------+----------+")
    print("| SAFE     | %4d     | %4d     |" % (hit_safe, miss_safe))
    print("+----------+----------+----------+")
    print("| PHISHING | %4d     | %4d     |" % (miss_safe, hit_phishing))
    print("+----------+----------+----------+")


def __main__():
    processor = DataProcessor()
    processor.build_dataset("dataset.csv")

    classifier = NeuralClassifier(processor.train_dataset)
    classifier.train()

    run_test(classifier, processor.test_dataset)


if __name__ == "__main__":
    __main__()
