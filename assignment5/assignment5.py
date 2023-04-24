import pandas as pd

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from keras.backend import clear_session

import warnings
warnings.filterwarnings('ignore')


def main():
    file_name = "spambase.data"
    df = pd.read_csv(file_name, header=None)
    data = df.to_numpy()

    train, test = train_test_split(data, test_size=0.2, random_state=1234)
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    model = Sequential()
    model.add(layers.Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dense(8, input_dim=X_train.shape[1], activation='relu'))
    # relu, tanh, softmax < sigmoid
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    model.fit(X_train, y_train,
              epochs=50,
              verbose=True,
              validation_data=(X_test, y_test),
              batch_size=10)

    clear_session()
    _, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print(f"Training Accuracy: {accuracy * 100:.4f}%")
    _, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print(f"Testing Accuracy:  {accuracy * 100:.4f}%")


if __name__ == "__main__":
    main()
