from pwc.classifier import Classifier
from pwc.classifier import Website
from pwc.classifier import remove_result
from pwc.classifier import dec_to_nparray
from sklearn.neural_network import MLPClassifier
import numpy as np


class NeuralClassifier(Classifier):

    __clf = MLPClassifier()

    def train(self):

        print("NeuralClassifier:   Training data ...")

        x = []
        y = []

        for site in self.train_dataset:
            x.append(list(remove_result(site.attributes).values()))
            y.append(site.attributes[Website.Tag.RESULT.value])

        self.__clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.__clf.fit(x, y)

        print("NeuralClassifier:   Data trained!")

    def classify(self, site):
        print("NeuralClassifier:   Classifying website ...")
        p = self.__clf.predict([list(remove_result(site.attributes).values())])
        print("NeuralClassifier:   Website classified!")
        return p
