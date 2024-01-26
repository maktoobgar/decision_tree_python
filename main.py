from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tree import *

data_iris = load_iris()
data, target = data_iris.data, data_iris.target
train, test, train_output, test_output = train_test_split(
    data, target, test_size=0.4, random_state=45449
)


def main():
    tree = train_tree(train, train_output)
    print(
        f"Trained on {len(train)} Data.\nTested on {len(test)} Data.\nAccuracy: {round(tree.test(test, test_output), 2)}%\nVisual Representation:\n{tree}"
    )


if __name__ == "__main__":
    main()
