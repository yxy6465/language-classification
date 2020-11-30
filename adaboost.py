import tree
from features_collection import Entry
from weights import Weights

import math
import pickle

"""
File: adaboost.py
Author: Lyan Ye (yxy6465)
Description: To train the adaboost model for prediction. 
"""

# the maximum size of stumps
SIZE = 5


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


class Adaboost:
    """
    A class represents Adaboost model.
    """
    def __init__(self, train_file="./data/train.dat", test_file="./data/test.dat", out_file="./out/ensemble.o"):
        """
        Constructor for the adaboost model.
        :param train_file: the file to be train
        :param test_file:  the file to be test
        :param out_file:  the file that the trained model to be store
        """
        # the tree object
        self.tree = None
        # the file that store the trained model
        self.out_file = out_file
        # the list of the training data
        self.train_file = parse_file(train_file)
        # list of data will be tested
        self.test_file = parse_file(test_file)
        # to store each stump
        self.ensemble = []

    def train(self, ensemble_size=SIZE):
        """
        To train the model using Adaboost algorithm.
        :param ensemble_size: the size of the stumps
        :return:
        """
        entries = self.train_file
        features = set(entries[0].features.keys())
        weights = Weights(entries)
        self.ensemble = []

        # create and store each stump
        for i in range(ensemble_size):
            stump = tree.make_tree(entries, features, [], 1)
            error = 0

            for entry in entries:
                decision = stump.decide_classification(entry)
                if decision != entry.target:
                    error += entry.weight

            for j in range(len(entries)):
                entry = entries[j]
                decision = stump.decide_classification(entry)
                if decision == entry.target:
                    new_weight = entry.weight * error / (weights.total - error)
                    weights.update_weight(j, new_weight)

            weights.normalization()
            stump.weight = math.log(weights.total - error) / error
            self.ensemble.append(stump)

        # store the model to a binary file
        file = open(self.out_file, "wb")
        pickle.dump(self, file)
        file.close()

    def test(self, test_file):
        """
        To predict each line with label(en/nl) for the testing data file.
        :param test_file: the file to be tested
        :return:
        """
        # if no model is generated, use default data to train a tree
        if not self.tree:
            self.train()

        entries = parse_file(test_file)
        for entry in entries:
            prediction = self.classify(entry)
            print(prediction)

    def classify(self, entry):
        """
        To classify and find out the best classification feature.
        :param entry: the line
        :return: the best feature
        """
        nums = {}
        best_classification = None
        best_num = 0

        for stump in self.ensemble:
            decision = stump.decide_classification(entry)
            if decision in nums:
                nums[decision] += stump.weight
            else:
                nums[decision] = stump.weight

            if nums[decision] > best_num:
                best_num = nums[decision]
                best_classification = decision

        return best_classification

