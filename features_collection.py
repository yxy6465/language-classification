"""
File: features_collection.py
Author: Lyan Ye (yxy6465)
Description: A file for creating the Entry class that initialize and store information for
             each sentence from the training file.
"""


class Entry:
    """
    A class to represent an entry of the training data(sentences).
    """
    def __init__(self, line, label=False):
        # dictionary to store if the entry satisfies the features
        self.features = features_stat(line)
        # weight of the entry
        self.weight = None
        if label:
            # en/nl
            self.target = None
            # store the sentence itself
            self.line = line
        else:
            self.target = line[:2]
            self.line = line[3:]


def features_stat(line):
    """
    1.	Boolean: usage of the word “and” in daily English sentences
    2.	Boolean: usage of the word “en” in daily Dutch sentences
    3.	Boolean: usage of the word “the” in daily English sentences
    4.	Boolean: usage of the word “de” in daily Dutch sentences
    5.	Boolean: usage of the word “enn” in daily Dutch sentences
    6.	Boolean: usage of the word “het” in daily Dutch sentences
    7.	Boolean: contains the substring of “ij” in daily Dutch words
    8.	Range: Words in Dutch tend to be longer than words in English
    9.	Range: Frequency of the usage of double vowels (consecutive two same vowels) in Dutch is more than in English
    10.	Range: Frequency of the usage of double consonants (consecutive two same consonants) in Dutch is more than in English
    11.	Range: Frequency of letters in words such as “j, k, v, z” in Dutch is more than in English

    :param line:
    :return:
    """
    num_vowels, nums_consonants = find_consecutive_letter(line)
    words = set(line.lower().split())

    stat = {"has_and": "and" in words,
            "has_en": "en" in words,
            "has_the": "the" in words,
            "has_de": "de" in words,
            "has_enn": "enn" in words,
            "has_het": "het" in words,
            "has_ij": "ij" in line,
            "word_len": average_word_len(line),
            "double_vowels": num_vowels,
            "double_consonants": nums_consonants,
            "rares": frequency_rares(line)}
    return stat


def average_word_len(line):
    """
    Calculate the average length of words for the line.
    range1: 0-5
    range2: 6-8
    range3: 9-inf
    :param line: the line
    :return: the range that the average length of words for the line
    """
    range1 = (0, 5)
    range2 = (6, 9)
    range3 = (9, None)
    total_len = 0

    line = line.strip().split()
    for word in line:
        total_len += len(word)
    average = total_len // len(line)

    if average <= range1[1]:
        return range1
    if range2[0] <= average <= range2[1]:
        return range2
    return range3


def find_consecutive_letter(line):
    """
    count the number of consecutive letters of the line.
    :param line: the sentence
    :return: the range of the number of consecutive vowels and consecutive consonants
    """
    list_vowels = ["a", "e", "i", "o", "u"]

    vowels_count = 0
    consonants_count = 0

    for i in range(len(line) - 1):

        letter = line[i]
        next_letter = line[i+1]
        if letter in list_vowels and next_letter == letter:
            vowels_count += 1
        elif letter == next_letter:
            consonants_count += 1

    return define_range(vowels_count), define_range(consonants_count)


def define_range(value):
    """
    Create the range for classification.
    :param value: the value
    :return: the range that the value is in
    """
    range1 = (0, 3)
    range2 = (4, 7)
    range3 = (8, None)

    if value <= range1[1]:
        return range1
    if range2[0] <= value <= range2[1]:
        return range2
    return range3


def frequency_rares(line):
    """
    Count the number of the rare letter "j", "k", "v", "z" in the line.
    :param line: the sentence
    :return: the range of the number of the rare letter is in
    """
    rares = ["j", "k", "v", "z"]
    num = 0
    range1 = (0, 3)
    range2 = (4, 6)
    range3 = (7, 8)
    range4 = (9, None)
    for c in line:
        if c in rares:
            num += 1

    if num <= range1[1]:
        return range1
    if range2[0] <= num <= range2[1]:
        return range2
    if range3[0] <= num <= range3[1]:
        return range3

    return range4
