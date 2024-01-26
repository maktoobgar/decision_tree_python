import numpy as np

Reset = "\033[0m"
Red = "\033[31m"
Green = "\033[32m"
Yellow = "\033[33m"
Blue = "\033[34m"
Purple = "\033[35m"
Cyan = "\033[36m"
Orange = "\033[33m"
Gray = "\033[37m"
White = "\033[97m"

Colors = [Green, Blue, Purple, Cyan, Orange, Gray, Yellow]


class Decision:
    splits = None
    outputs = None
    middle = 0.0
    attr = None
    entropy = None
    length: int = 0

    def __init__(self, splits, outputs, attr, middle):
        self.outputs = outputs
        self.attr = attr
        self.splits = splits
        self.middle = middle
        for output in outputs:
            self.length += len(output)
        self.__entropy()

    def next(self, input) -> int:
        return 1 if input[self.attr] >= self.middle else 0

    def create_nodes(self) -> []:
        nodes = []
        for i in range(len(self.outputs)):
            nodes.append(Node(np.array(self.splits[i]), np.array(self.outputs[i])))
        return nodes

    def __entropy(self):
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


class Node:
    data: [] = []
    outputs: [] = []
    children: [] = []
    prediction: object = None
    decision: Decision = None
    depth: int = 0

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def best_decision(self, decisions) -> Decision:
        min_entropy = None
        for i in range(len(decisions)):
            if min_entropy == None or decisions[i].entropy < min_entropy.entropy:
                min_entropy = decisions[i]
        return min_entropy

    def expand(self, depth: int = 0) -> None:
        self.depth = depth
        if len(np.unique(self.outputs, return_counts=True)[1]) == 1:
            self.prediction = self.outputs[0]
            return
        attr_decisions = []
        for i in range(self.inputs.shape[1]):
            sorted_indices = np.argsort(self.inputs[:, i], axis=0)
            sorted_data = self.inputs[sorted_indices, :]
            outputs = np.array(self.outputs[sorted_indices])
            decisions = []
            for j in range(len(self.inputs)):
                if len(sorted_data[:j]) == 0 or len(sorted_data[:j]) == 0:
                    continue

                des = Decision(
                    [sorted_data[:j].tolist(), sorted_data[j:].tolist()],
                    [outputs[:j].tolist(), outputs[j:].tolist()],
                    i,
                    sorted_data[j][i],
                )
                decisions.append(des)
            decision = self.best_decision(decisions)
            attr_decisions.append(decision)
        self.decision = self.best_decision(attr_decisions)
        self.children = self.decision.create_nodes()
        for node in self.children:
            node.expand(depth + 1)

    def test(self, inputs, outputs) -> float:
        truth = 0
        for i in range(len(inputs)):
            input = inputs[i]
            output = outputs[i]
            node = self
            while node.decision:
                node = node.children[node.decision.next(input)]
            if node.prediction == output:
                truth += 1
        return truth / len(inputs) * 100

    def __str__(self) -> str:
        if len(self.children) > 0:
            output = f"{Colors[self.depth]} <-- {self.decision.middle} < ({self.depth}-on {self.decision.attr}) <= {self.decision.middle} --> {Reset}"
            return f"{Colors[self.depth + 1]}({Reset}{self.children[0]}{Colors[self.depth + 1]}){Reset}{output}{Colors[self.depth+1]}({Reset}{self.children[1]}{Colors[self.depth + 1]}){Reset}"
        return f"{Red}{self.prediction}{Reset}"


def train_tree(inputs, outputs) -> Node:
    n = Node(inputs, outputs)
    n.expand()
    return n
