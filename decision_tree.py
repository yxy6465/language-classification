import tree
from features_collection import Entry

import pickle

"""
File: tree.py
Author: Lyan Ye (yxy6465)
Description: To train the decision tree model and use it to predict the testing data.
"""


# the maximum depth of the decision tree
MAX_DEPTH = 15


def parse_file(filename):
    """
    To parse the file and store each line to the list
    :param filename:
    :return: the list of entries
    """
    entries = []
    file = open(filename, encoding="utf-8")
    for line in file:
        entries.append(Entry(line))
    return entries


class DecisionTree:
    """
    A class for decision tree model.
    """
    def __init__(self, train_file="./data/train.dat", test_file="./data/test.dat", out_file="./out/tree.o"):
        """
        Constructor for the decision tree model
        :param train_file: the file to be trained
        :param test_file: the file to be tested
        :param out_file: the file to store the model
        """
        # the tree object
        self.tree = None
        # the file that store the trained model
        self.out_file = out_file
        # the list of the training data
        self.train_file = parse_file(train_file)
        # the list of the testing data
        self.test_file = parse_file(test_file)

    def train(self):
        """
        To train the model using decision tree algorithm.
        :return:
        """
        entries = self.train_file
        features = set(entries[0].features.keys())

        self.tree = tree.make_tree(entries, features, [], MAX_DEPTH)

        file = open(self.out_file, "wb")
        pickle.dump(self, file)
        file.close()

    def test(self, test_file):
        """
        To predict the testing data using decision tree model.
        :param test_file: the file to be tested
        :return:
        """
        if not self.tree:
            self.train()
        entries = parse_file(test_file)
        for entry in entries:
            prediction = self.tree.decide_classification(entry)
            print(prediction)
