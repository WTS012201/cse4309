# William Sigala
# ID: 1001730022

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')


class Node:
    def __init__(
        self,
        left=None,
        right=None,
        feature=None,
        threshold=None,
        value=None
    ):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value


class DecisionTree:
    def __init__(self, feature_names, max_depth=10):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.root = None

    def _compute_entropy(self, y):
        proportions = np.array([np.count_nonzero(y == val)
                                for val in self.label_counter]) / len(y)

        entropy = 0
        for val in proportions:
            if val > 0:
                entropy += val * np.log2(val)

        return -entropy

    def _compute_info_gain(self, X, y, thresh):
        loss = self._compute_entropy(y)
        left = np.argwhere(X <= thresh).flatten()
        right = np.argwhere(X > thresh).flatten()
        n, n_left, n_right = len(y), len(left), len(right)

        if not n_left or not n_right:
            return 0

        child_loss = (n_left / n) * self._compute_entropy(y[left]) \
            + (n_right / n) * self._compute_entropy(y[right])
        return loss - child_loss

    def _best(self, X, y, features):
        max_it = 1000
        score, feature, thresh = None, None, None

        for _feat in features:
            samples = X[:, _feat]
            iters = np.unique(samples)
            for _thresh in iters[:max_it] if len(iters) > max_it else iters:
                _score = self._compute_info_gain(samples, y, _thresh)

                if score is None or _score > score:
                    score, feature, thresh = _score, _feat, _thresh

        return feature, thresh

    def _generate_node(self, X, y, depth=0, condition="<="):
        self.n_samples, self.n_features = X.shape

        self.label_counter = {}
        self.label_counter = {
            val: 1 if val not in self.label_counter else
            self.label_counter[val] + 1 for val in y}
        max = None
        for feature, occurences in self.label_counter.items():
            if max is None or occurences > max[1]:
                max = (feature, occurences)
        self.n_class_labels = len(self.label_counter)

        if (depth >= self.max_depth
                or self.n_class_labels == 1):

            return Node(value=np.int64(max[0]))

        features = np.random.choice(
            self.n_features, self.n_features, replace=False)
        feature, threshold = self._best(X, y, features)
        self._print_node(
            depth, self.feature_names[feature], f"{condition} {threshold:.2f}")

        samples = X[:, feature]
        left = np.argwhere(samples <= threshold).flatten()
        right = np.argwhere(samples > threshold).flatten()

        left = self._generate_node(X[left, :], y[left], depth + 1, "<=")
        right = self._generate_node(
            X[right, :], y[right], depth + 1, ">")
        return Node(left, right, feature, threshold)

    def _gen_predict(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._gen_predict(x, node.left)
        return self._gen_predict(x, node.right)

    def _print_node(self, depth, feature_name, condition):
        print(f"{depth + 1} ", end="")
        for _ in range(depth):
            print("|   ", end="")
        print(f"|--- {feature_name} {condition}")

    def fit(self, X, y):
        self.root = self._generate_node(X, y)

    def predict(self, X):
        predictions = [self._gen_predict(x, self.root) for x in X]
        return np.array(predictions)


def load_data(file):
    train = pd.read_csv(file)
    train = train.replace("?", np.nan)
    train = train.fillna(train.median()).astype(np.float64)
    train = shuffle(train)
    X, Y = train.iloc[:, :-1], train.iloc[:, -1]

    return train.columns, X.to_numpy(), Y.to_numpy(dtype="int64")


def validation(model):
    _, X, Y = load_data("bvalidate.csv")
    n = Y.shape[0]
    correct = (model.predict(X) == Y).sum()
    print(f"\nValidation accuracy: {(correct / n * 100):.2f} ")


def test(model):
    feature_names, X, _ = load_data("btest.csv")

    df = pd.DataFrame(X, columns=feature_names[:-1])
    df["winner"] = model.predict(X)

    df.to_csv("output.csv")


def main():
    feature_names, X, Y = load_data("btrain.csv")

    clf = DecisionTree(feature_names, max_depth=5)
    clf.fit(X, Y)

    validation(clf)
    test(clf)


if __name__ == "__main__":
    main()
