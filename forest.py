import math
import random

import numpy as np

from tree import train_tree


def train_forest(data: [], outputs: [], trees_count: int, n: int) -> []:
    trees = []
    attributes_len = len(data[0])
    dbs = []
    db_outputs = []
    for _ in range(trees_count):
        db = []
        db_output = []
        for _ in range(n):
            index = random.randrange(0, len(data))
            db.append(data[index])
            db_output.append(outputs[index])
        db = np.array(db)
        db_output = np.array(db_output)
        db_outputs.append(db_output)
        dbs.append(db)
    db_outputs = np.array(db_outputs)
    dbs = np.array(dbs)
    for i in range(len(dbs)):
        db = dbs[i]
        db_output = db_outputs[i]
        chosen_attrs = []
        for _ in range(int(math.log2(attributes_len + 1))):
            while True:
                index = random.randrange(0, attributes_len)
                if index in chosen_attrs:
                    continue
                chosen_attrs.append(index)
                break
        for i in range(len(data[0])):
            if i not in chosen_attrs:
                db[:, i] = 0
        trees.append(train_tree(db, db_output))

    return trees


def test_forest(forest: [], test: [], test_output: []) -> float:
    predictions = []
    for i in range(len(forest)):
        predictions.append(forest[i].predict(test))
    predictions = np.array(predictions)
    predictions_per_each_data = []
    for i in range(len(test)):
        data = predictions[:, i]
        data.sort()
        predictions_per_each_data.append(data)
    predictions_per_each_data = np.array(predictions_per_each_data)
    truth = 0
    for i in range(len(test)):
        median = np.median(predictions_per_each_data[i])
        if median == test_output[i]:
            truth += 1
    return truth / len(test) * 100
