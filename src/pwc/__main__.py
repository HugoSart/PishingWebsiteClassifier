from pwc.classifier import DataProcessor, Classifier
from pwc.classifier import Website
from pwc.neural_classifier import NeuralClassifier
from pwc.svm_classifier import SvmClassifier
import argparse


def run_single(classifier):

    d = dict()
    print                                                    ("+---------------------------------------+-----------------+--------+")
    print                                                    ("| Property description                  | Possible values | Value  |")
    print                                                    ("+---------------------------------------+-----------------+--------+")
    d[Website.Tag.IP.value]                     = float(input("  IP address instead of website name      [-1,    1]        "))
    d[Website.Tag.URL_LENGTH.value]             = float(input("  Long URL to hide suspicious part        [-1, 0, 1]        "))
    d[Website.Tag.SHORT_SERVICE.value]          = float(input("  Using URL shortening service            [-1,    1]        "))
    d[Website.Tag.SYMBOL.value]                 = float(input("  Having at Symbol                        [-1,    1]        "))
    d[Website.Tag.DOUBLE_SLASH_REDIRECT.value]  = float(input("  Using double slash redirecting          [-1,    1]        "))
    d[Website.Tag.PREFIX_SUFFIX.value]          = float(input("  Uses prefix or suffix separated by [-]  [-1,    1]        "))
    d[Website.Tag.SUB_DOMAIN.value]             = float(input("  Has sub domain                          [-1,    1]        "))
    d[Website.Tag.SSL.value]                    = float(input("  Final state is SSL                      [-1, 0, 1]        "))
    d[Website.Tag.DOMAIN_LENGTH.value]          = float(input("  Domain registration length              [-1,    1]        "))
    d[Website.Tag.FAVICON.value]                = float(input("  Favicon                                 [-1,    1]        "))
    d[Website.Tag.PORT.value]                   = float(input("  Using non-standard port                 [-1,    1]        "))
    d[Website.Tag.TOKEN.value]                  = float(input("  HTTPS Token in domain part of URL       [-1,    1]        "))
    d[Website.Tag.REQUEST.value]                = float(input("  Request URL                             [-1,    1]        "))
    d[Website.Tag.URL_ANCHOR.value]             = float(input("  URL of anchor                           [-1, 0, 1]        "))
    d[Website.Tag.LINK_IN_TAB.value]            = float(input("  Links in tags                           [-1, 0, 1]        "))
    d[Website.Tag.SFH.value]                    = float(input("  SFH with empty string or about:blank    [-1, 0, 1]        "))
    d[Website.Tag.EMAIL.value]                  = float(input("  Submitting information to email         [-1,    1]        "))
    d[Website.Tag.ABNORMAL_URL.value]           = float(input("  Abnormal URL                            [-1,    1]        "))
    d[Website.Tag.REDIRECT.value]               = float(input("  Number of redirects (<= 1 is good)      [    0, 1]        "))
    d[Website.Tag.MOUSE_OVER.value]             = float(input("  onMouseHover change status bar (source) [-1,    1]        "))
    d[Website.Tag.RIGHT_CLICK.value]            = float(input("  Right click is disabled                 [-1,    1]        "))
    d[Website.Tag.POPUP.value]                  = float(input("  Has popup window                        [-1,    1]        "))
    d[Website.Tag.IFRAME.value]                 = float(input("  IFrame redirection                      [-1,    1]        "))
    d[Website.Tag.AGE.value]                    = float(input("  Domain older than 6 months              [-1,    1]        "))
    d[Website.Tag.DNS.value]                    = float(input("  Domain DNS record for is not empty      [-1,    1]        "))
    d[Website.Tag.TRAFFIC.value]                = float(input("  Website pop. in Alexa is > 100.000      [-1,    1]        "))
    d[Website.Tag.RANK.value]                   = float(input("  Less than 0.2 points in PageRank        [-1,    1]        "))
    d[Website.Tag.INDEX.value]                  = float(input("  Indexed by Google                       [-1,    1]        "))
    d[Website.Tag.REFERENCES.value]             = float(input("  Links pointing to page (0 bad, 2 good)  [-1, 0, 1]        "))
    d[Website.Tag.REPORTS.value]                = float(input("  Belongs to Top Phishing                 [-1,    1]        "))
    print                                                    ("  Result                                  [-1,    1]        ", classifier.classify(Website(d)))


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

    print("\n======================== TEST RESULTS =========================\n")
    print("  +----------+----------+----------+    Precision  :  %4.2f %%" % precision)
    print("  |          | SAFE     | PHISHING |")
    print("  +----------+----------+----------+    Recall     :  %4.2f %%" % recall)
    print("  | SAFE     | %4d     | %4d     |" % (hit_safe, miss_safe))
    print("  +----------+----------+----------+    F-Measure  :  %4.2f %%" % f)
    print("  | PHISHING | %4d     | %4d     |" % (miss_phishing, hit_phishing))
    print("  +----------+----------+----------+    Accuracy   :  %4.2f %%" % accuracy)
    print("\n===============================================================")


def __main__(classifier_str,  mode_str):

    processor = DataProcessor()
    processor.build_dataset("dataset.csv")

    classifier = Classifier
    if classifier_str == "svm":
        classifier = SvmClassifier(processor.train_dataset)
    elif classifier_str == "neural":
        classifier = NeuralClassifier(processor.train_dataset)
    else:
        print("Invalid classifier")
        exit(-1)

    classifier.train()

    if mode_str == "test":
        run_test(classifier, processor.test_dataset)
    elif mode_str == "single":
        run_single(classifier)
    else:
        print("Invalid mode")
        exit(-1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier", type=str, help="The classifier method to be used [smv | default: neural]")
    parser.add_argument("-m", "--mode", type=str, help="Application mode [test | default: single]")
    args = parser.parse_args()

    if args.classifier is None:
        args.classifier = "neural"
    if args.mode is None:
        args.mode = "test"

    __main__(args.classifier, args.mode)
