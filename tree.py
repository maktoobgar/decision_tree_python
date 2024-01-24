import math
import numpy as np


class Decision:
    splits: []
    outputs: []
    attribute: int
    entrpy: float
    length: int

    def __init__(self, splits: [], outputs: [], attribute: int) -> None:
        self.splits = splits
        self.outputs = outputs
        self.attribute = attribute
        self.length = np.sum([len(output) for output in outputs])
        self.__entropy()

    def __entropy(self) -> None:
        each_side_entropies = []
        for outputs in self.outputs:
            proabilities = np.unique(outputs, return_counts=True)[1] / len(outputs)
            each_side_entropies.append(
                np.sum([p * np.log2(1 / p) for p in proabilities])
            )
        self.entropy = np.sum(
            [
                each_side_entropies[i] * len(self.outputs[i]) / self.length
                for i in range(len(each_side_entropies))
            ]
        )

        pass

    def create_nodes(self) -> []:
        pass


class Node:
    inputs: []
    outputs: []
    decision: Decision
    next_nodes: []

    def __init__(self, inputs: [], outputs: []) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def calculate_decision(self) -> Decision:
        pass

    def create_next_nodes(self) -> None:
        pass


def train_tree(inputs: [], outputs: []) -> Node:
    pass
