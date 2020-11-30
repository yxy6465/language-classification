class Weights:
    """
    A class for representing the weighted entry(each sentence).
    """
    def __init__(self, entries):
        """
        Constructor of the Weights class.
        :param entries: list of entries(sentences)
        """
        self.entries = entries
        self.total = 0

        for entry in self.entries:
            entry.weight = 1
            self.total += entry.weight

        self.sum = self.total

    def size(self):
        """
        Size of the list of entries
        :return: length of the list of entries
        """
        return len(self.entries)

    def update_weight(self, index, new_weight):
        """
        Update the weight of the specific entry
        :param index: the index of the specific entry in the list
        :param new_weight: the new weight value to be set
        :return:
        """
        self.total -= self.entries[index].weight
        self.entries[index].weight = new_weight
        self.total += new_weight

    def normalization(self):
        """
        Normalization using weight of each entry
        :return:
        """
        z_score = self.sum / self.total
        self.total = 0
        for entry in self.entries:
            entry.weight = entry.weight * z_score
            self.total += entry.weight
