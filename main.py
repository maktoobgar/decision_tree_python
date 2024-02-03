import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from forest import test_forest, train_forest
from tree import train_tree

data_iris = load_iris()
data, target = data_iris.data, data_iris.target
train, test, train_output, test_output = train_test_split(
    data, target, test_size=0.4, random_state=44
)


def main():
    tree = train_tree(train, train_output)
    print(
        f"Trained on {len(train)} Data.\nTested on {len(test)} Data.\nAccuracy: {round(tree.test(test, test_output), 2)}%\nVisual Representation:\n{tree}"
    )


def forest():
    forest = train_forest(train, train_output, 50, len(train) - 10)
    print(
        f"Trained on {len(train)} Data.\nTested on {len(test)} Data.\nAccuracy: {round(test_forest(forest, test, test_output), 2)}%"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "forest":
        forest()
    else:
        main()
