from decision_tree import DecisionTree
from adaboost import Adaboost

import os
import sys
import pickle

"""
File: classify.py
Author: Lyan Ye (yxy6465)
Description: The main file for run the program to train the model or predict the 
             test data file.
"""

# constants
ENGLISH = "en"
DUTCH = "nl"

ADABOOST = "ada"
DECISION_TREE = "dt"


def print_instruction(training = True, prediction = True):
    """
    To print out the instruction options to run the program.
    :param training: if print command for action of training
    :param prediction: if print command for action of prediction
    :return:
    """
    if training:
        print("To train: python classify.py train <examples> <hypothesisOut> <learning-type>")
    if prediction:
        print("To predict: python classify.py predict <hypothesis> <file>")


def train(examples, hypothesis_out, algo):
    """
    To train the model according to the user's choice.
    :param examples: the training data
    :param hypothesis_out: the file to store the model
    :param algo: the algorithm that the user choose. dt stands for decision tree,
                 ada stands for adaboost
    :return:
    """
    if algo == DECISION_TREE:
        model = DecisionTree(train_file=examples, out_file=hypothesis_out)

    elif algo == ADABOOST:
        model = Adaboost(train_file=examples, out_file=hypothesis_out)

    else:
        print("Please use \"dt\" as decision tree model, \"ada\" as Adaboost model")
        sys.exit()
    model.train()


def predict(hypothesis, test_file):
    """
    To predict the testing data.
    :param hypothesis: the file stores the model
    :param test_file: the file to be tested
    :return:
    """
    file = open(hypothesis, "rb")
    model = pickle.load(file)
    model.test(test_file)
    file.close()


def main():
    """
    Main function to setup and run the program.
    :return:
    """
    if len(sys.argv) < 2:
        print_instruction()
    what_to_do = sys.argv[1]
    if what_to_do == "train":
        if len(sys.argv) != 5:
            print_instruction(prediction=False)
        examples = sys.argv[2]
        hypothesis_out = sys.argv[3]
        algo = sys.argv[4]
        print("Start the training process...")
        train(examples, hypothesis_out, algo)
        print("Training process is done.")

    elif what_to_do == "predict":
        if len(sys.argv) != 4:
            print_instruction(training=False)
        hypothesis = sys.argv[2]
        test_file = sys.argv[3]
        predict(hypothesis, test_file)

    else:
        print_instruction()
        sys.exit()


if __name__ == "__main__":
    main()
