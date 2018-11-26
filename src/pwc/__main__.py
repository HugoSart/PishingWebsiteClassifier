from pwc.classifier import DataProcessor
from pwc.classifier import Website
from pwc.neural_classifier import NeuralClassifier


def run_test(classifier, test_dataset):

    hit_safe = miss_safe = hit_phishing = miss_phishing = 0

    total = total_safe = total_phishing = 0
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

    total_safe = hit_safe + miss_phishing
    total_phishing = hit_phishing + miss_safe
    total = total_safe + total_phishing

    precision = (hit_safe / (hit_safe + miss_safe)) * 100
    recall = (hit_safe / (hit_safe + miss_phishing)) * 100
    f = 2 * ((precision * recall) / (precision + recall))
    accuracy = ((hit_safe + hit_phishing) / total) * 100

    print("\n======================== TEST RESULTS ========================\n")
    print("  +----------+----------+----------+    Precision  : %4.2f %%" % precision)
    print("  |          | SAFE     | PHISHING |")
    print("  +----------+----------+----------+    Recall     : %4.2f %%" % recall)
    print("  | SAFE     | %4d     | %4d     |" % (hit_safe, miss_safe))
    print("  +----------+----------+----------+    F-Measure  : %4.2f %%" % f)
    print("  | PHISHING | %4d     | %4d     |" % (miss_phishing, hit_phishing))
    print("  +----------+----------+----------+    Accuracy   : %4.2f %%" % accuracy)
    print("\n==============================================================")


def __main__():
    processor = DataProcessor()
    processor.build_dataset("dataset.csv")

    classifier = NeuralClassifier(processor.train_dataset)
    classifier.train()

    run_test(classifier, processor.test_dataset)


if __name__ == "__main__":
    __main__()
