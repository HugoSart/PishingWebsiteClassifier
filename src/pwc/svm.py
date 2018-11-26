from sklearn import svm
from pwc.classifier import Classifier, Website, remove_result


class SvmClassifier(Classifier):

    classifier = svm.SVC(gamma='scale')

    def train(self):
        labels = []
        attributes = []
        for x in self.train_dataset:
            attributes.append(list(remove_result(x.attributes).values()))
            labels.append(x.attributes[Website.Tag.RESULT.value])
        print(attributes)
        self.classifier.fit(attributes, labels)
        return

    def classify(self, site):
        return self.classifier.predict([list(remove_result(site.attributes).values())])[0]