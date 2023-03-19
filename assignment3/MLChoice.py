# William Sigala
# ID: 1001730022

# sample inputs:
# python MLChoice.py knn banknote
# python MLChoice.py svm sonar

import sys
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
        self.choice = choice
        self.train, self.test = self.load_data()

        if choice == "knn":
            k = 5
            self.predict = lambda x: self.knn(x, k)
            self.skl = skl_knn(n_neighbors=k)
        elif choice == "svm":
            n_iter = 1000
            self.predict = lambda x: self.svm(x, n_iter=n_iter)
            self.skl = skl_svm(kernel="linear")
            self.skl.n_iter = n_iter

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

        # for binary classification svm
        group1, group2 = [*train_set.keys()]
        train_set = {
            (-1, group1): train_set[group1],
            (1, group2): train_set[group2],
        }
        test_set = {
            (-1, group1): test_set[group1],
            (1, group2): test_set[group2],
        }
        self.class_map = {k: v for (k, v) in train_set.keys()}
        self.X, self.y = [], []
        for group, group_set in train_set.items():
            self.X += group_set
            self.y += [group[0]] * len(group_set)

        return train_set, test_set

    def log_results(self):
        ds = "Bank Note" if self.dataset == "banknote" else "Sonar"
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
        sample_i = np.random.randint(0, len(self.test[sample_group]))
        sample = self.test[sample_group][sample_i]

        prediction = self.predict(sample)

        return sample, self.class_map[prediction], sample_group[1]

    def classify(self):
        skl_correct, correct, total = 0, 0, 0
        self.skl.fit(self.X, self.y)

        for group in self.test:
            for data in self.test[group]:
                vote = self.predict(data)
                skl_vote = self.skl.predict([data])
                if group[0] == vote:
                    correct += 1
                if group[0] == skl_vote[0]:
                    skl_correct += 1
                total += 1

        self.accuracy = round(correct / total, 4) * 100
        self.skl_accuracy = round(skl_correct / total, 4) * 100

    def knn(self, predict, k):
        distances = []
        for group in self.train:
            for features in self.train[group]:
                euclidean_distance = np.linalg.norm(
                    np.array(features) - np.array(predict)
                )
                distances.append([euclidean_distance, group])
        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]

        return vote_result[0]

    def svm(self, predict, n_iter):
        predict = np.array(predict)
        result = None

        try:
            result = np.sign(np.dot(predict, self.w) - self.b)
        except AttributeError:
            X, y = np.array(self.X),  np.array(self.y)
            n_features = X.shape[1]

            y = np.where(y <= 0, -1, 1)

            self.w = np.zeros(n_features)
            self.b = 0
            lr = 0.001
            _lambda = 0.01

            for _ in range(n_iter):
                for i, x in enumerate(X):
                    if y[i] * (np.dot(x, self.w) - self.b) >= 1:
                        self.w -= lr * (2 * _lambda * self.w)
                    else:
                        self.w -= lr * (
                            2 * _lambda * self.w - np.dot(x, y[i])
                        )
                        self.b -= lr * y[i]

            result = np.sign(np.dot(predict, self.w) - self.b)

        return result


def main():
    choice, dataset = sys.argv[1].lower(), sys.argv[2].lower()
    MLChoice(choice, dataset)


if __name__ == "__main__":
    main()
