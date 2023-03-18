# William Sigala
# ID: 1001730022

# sample inputs:
# python MLChoice.py knn banknote
# python MLChoice.py svm sonar

import sys
import warnings
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier as skl_knn
from sklearn.svm import SVC as skl_svm

BANKNOTE_DATA_FILE = "data_banknote_authentication.txt"
SONAR_DATA_FILE = "sonar.txt"


class MLChoice:
    def __init__(self, choice, dataset):
        self.dataset = dataset
        self.train, self.test = self.load_data()
        self.choice = choice

        if choice == "knn":
            k = 5
            self.predict = lambda x: self.knn(x, k)
            self.skl = skl_knn(n_neighbors=k)
        elif choice == "svm":
            self.predict = self.svm
            self.skl = skl_svm(kernel="linear", probability=True)

        self.result = self.classify()
        self.log_results()

    def load_data(self):
        file = BANKNOTE_DATA_FILE if self.dataset == "banknote" else SONAR_DATA_FILE
        data = np.array(pd.read_csv(file, header=None).values)
        data = shuffle(data)
        test_size = 0.2

        train_data = data[:-int(test_size*len(data))]
        test_data = data[-int(test_size*len(data)):]

        train_set, test_set = {}, {}
        for i in train_data:
            if i[-1] in train_set:
                train_set[i[-1]].append(i[:-1].astype(float))
            else:
                train_set[i[-1]] = [i[:-1].astype(float)]

        for i in test_data:
            if i[-1] in test_set:
                test_set[i[-1]].append(i[:-1].astype(float))
            else:
                test_set[i[-1]] = [i[:-1].astype(float)]

        self.X, self.Y = [], []
        for group in train_set:
            for data in train_set[group]:
                self.X.append(data)
                self.Y.append(group)

        return train_set, test_set

    def log_results(self):
        ds = "Banke Note" if self.dataset == "banknote" else "Sonar"
        print(
            f"DataSet: {ds}\n\n" +
            f"Machine Learning Algorithm Chosen: {self.choice.upper()}\n\n" +
            f"Accuracy of Training (Scratch): {self.accuracy:.2f}%\n\n" +
            f"Accuracy of ScikitLearn Function: {self.skl_accuracy:.2f}%\n"
        )

        sample, prediction, actual = self.random_test_sample()
        print(
            f"Prediction Point: {np.array(sample)}\n" +
            f"Predicted Class: {prediction}\n" +
            f"Actual class: {actual}\n"
        )

    def random_test_sample(self):
        groups = [*self.test.keys()]
        sample_group = groups[np.random.randint(0, len(groups))]
        sample_idx = np.random.randint(0, len(self.test[sample_group]))
        sample = self.test[sample_group][sample_idx]
        prediction = self.predict(sample)

        return sample, prediction, sample_group

    def classify(self):
        skl_correct, correct, total = 0, 0, 0
        self.skl.fit(self.X, self.Y)

        for group in self.test:
            for data in self.test[group]:
                vote = self.predict(data)
                skl_vote = self.skl.predict([data])
                if group == vote:
                    correct += 1
                if group == skl_vote:
                    skl_correct += 1
                total += 1

        self.accuracy = round(correct / total, 4) * 100
        self.skl_accuracy = round(skl_correct / total, 4) * 100
        return None

    def knn(self, predict, k):
        data = self.train
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')

        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(
                    np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])
        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result

    def svm(self, predict):
        return 0


def main():
    choice, dataset = sys.argv[1].lower(), sys.argv[2].lower()
    MLChoice(choice, dataset)


if __name__ == "__main__":
    main()
