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
from sklearn.neighbors import KNeighborsClassifier as KNN

BANKNOTE_DATA_FILE = "data_banknote_authentication.txt"
SONAR_DATA_FILE = "sonar.txt"


class MLChoice:
    def __init__(self, choice, dataset):
        self.dataset = dataset
        self.train, self.test = self.load_data()
        self.choice = choice
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

        return train_set, test_set

    def log_results(self):
        print(
            f"DataSet: {self.dataset}\n\n" +
            f"Machine Learning Algorithm Chosen: {self.choice.upper()}\n\n" +
            f"Accuracy of Training (Scratch): {self.accuracy}\n\n" +
            f"Accuracy of ScikitLearn Function: {self.skl_accuracy}\n\n"
        )

        return

    def classify(self):
        skl_correct, correct, total = 0, 0, 0
        skl_knn = KNN(n_neighbors=5)
        X, y = [], []
        for group in self.test:
            for data in self.test[group]:
                X.append(data)
                y.append(group)
        skl_knn.fit(X, y)

        for group in self.test:
            for data in self.test[group]:
                vote = self.knn(self.train, data, k=5)
                skl_vote = skl_knn.predict([data])
                if group == vote:
                    correct += 1
                if group == skl_vote:
                    skl_correct += 1
                total += 1

        self.accuracy = correct / total
        self.skl_accuracy = skl_correct / total
        return None

    def knn(self, data, predict, k=5):
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')

        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(
                    np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])
        # print(distances)
        votes = [i[1] for i in sorted(distances)[:k]]
        # print(votes)
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result

    def svm(self):
        pass


def main():
    choice, dataset = sys.argv[1].lower(), sys.argv[2].lower()
    MLChoice(choice, dataset)


if __name__ == "__main__":
    main()
