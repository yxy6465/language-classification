import math

"""
File: tree.py
Author: Lyan Ye (yxy6465)
Description: The tree to store all the information for classification.
"""

# maximum depth of tree
MAX_DEPTH = 15


def make_tree(entries, features, parent, depth):
    """
    To make the tree with Node class.
    :param entries: the training data
    :param features: the features list
    :param parent: the parent of this (sub)tree
    :param depth: the current depth
    :return: the tree
    """
    if not entries:
        return Node(get_plurality(parent), is_leaf=True)
    if check_same_value(entries):
        return Node(entries[0].target, is_leaf=True)
    if not features:
        return Node(get_plurality(entries), is_leaf=True)

    best_feature, children = get_max_gain(entries, features)
    root = Node(best_feature)

    # fix unknown error if any
    if depth < 1:
        depth = 1

    for value in children:
        # partial list after classified
        partial = children[value]
        if depth == 1:
            sub = Node(get_plurality(partial), is_leaf=True)
            root.add_child(value, sub)
        else:
            sub = make_tree(partial, features.difference({best_feature}), entries, depth - 1)
            root.add_child(value, sub)
    return root


class Node:
    """
    A class represents the tree with Node.
    """

    def __init__(self, value, is_leaf=False):
        """
        Constructor for the Node.
        :param value: value of the node(feature)
        :param is_leaf: True if the node is leaf
        """
        self.value = value  # feature
        self.is_leaf = is_leaf
        self.children = {}
        self.weight = None

    def add_child(self, label, node):
        """
        To add a child to the node
        :param label: the label of the node (range, boolean)
        :param node: the node itself
        :return:
        """
        self.children[label] = node

    def decide_classification(self, entry):
        """
        To classify the entry
        :param entry: representing the line
        :return: the classification
        """
        node = self
        while node:
            if node.is_leaf:
                return node.value

            child_type = entry.features[node.value]
            if child_type in node.children:
                node = node.children[child_type]
            else:
                return find_classification(node)

        return None


def find_classification(node):
    """
    Find the classification among the children nodes.
    :param node: the node to be looked for its children
    :return: the best classification from the children nodes
    """
    if node.is_leaf:
        return node.value

    if not node.children:
        return None

    nums = {}
    count = -1
    best_classification = None

    for child_type in node.children:
        classification = find_classification(node.children[child_type])

        if not classification:
            continue

        if classification in nums:
            nums[classification] += 1
        else:
            nums[classification] = 1

        if nums[classification] > count:
            count = nums[classification]
            best_classification = classification

    return best_classification


def num_entries(entries):
    """
    Find the number(weight) of each entry type(target)
    :param entries: the examples from the training data
    :return: the dictionary with entry type as key and counts as value.
    """
    nums = {}
    for e in entries:
        if e.weight:
            weight = e.weight
        else:
            weight = 1

        if e.target in nums:
            nums[e.target] += weight
        else:
            nums[e.target] = weight
    return nums


def get_plurality(entries):
    """
    Get the majority classification from the entries.
    :param entries: the list of entries
    :return: the classification
    """
    value = None
    weight = -1
    nums = num_entries(entries)
    # find the plurality entry type with the maximum weight
    for e in entries:
        if nums[e.target] > weight:
            weight = nums[e.target]
            value = e.target
    return value


def check_same_value(entries):
    """
    Check if all entry in the list has the same value(classification).
    :param entries: the list contains entries
    :return: True if all same, False otherwise.
    """
    target = entries[0].target
    for i in range(1, len(entries)):
        if entries[i] != target:
            return False
    return True


def get_entropy(entries):
    """
    To find the entropy value of current level.
    :param entries: the entries
    :return: the entropy of the current list
    """
    amounts = num_entries(entries)
    entropy = 0
    for k in amounts.keys():
        p = amounts[k]/len(entries)
        entropy += -1 * p * math.log(p, 2)

    return entropy


def classify(entries, feature):
    """
    To classify(split) the list of entries
    :param entries: the list of entries
    :param feature: the feature to classify
    :return: the classified children
    """
    children = {}
    for entry in entries:
        value = entry.features[feature]

        if value in children:
            children[value].append(entry)
        else:
            children[value] = [entry]

    return children


def get_information_gain(entries, feature, entropy):
    """
    Get the information gain value of the current level
    :param entries: the entries to be calculated
    :param feature: the feature used for classification
    :param entropy: the entropy value of current level
    :return: the information gain and classified children
    """
    children = classify(entries, feature)
    gain = 0
    for value in children:
        # partial classified children
        partial = children[value]
        gain += len(partial) * get_entropy(partial) / len(entries)
    return entropy - gain, children


def get_max_gain(entries, features):
    """
    Find the best (greatest) information gain value among features.
    :param entries: the list of entries
    :param features: the features list
    :return: best feature and the corresponding children
    """
    entropy = get_entropy(entries)
    best_feature = None  # best feature to classify
    best_children = None  # best classification children
    best_gain = -1  # best information gain

    for feature in features:
        gain, children = get_information_gain(entries, feature, entropy)

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_children = children

    return best_feature, best_children

