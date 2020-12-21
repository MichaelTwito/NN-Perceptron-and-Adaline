import random

import numpy


class handler(object):
    """
    This class will split the two sets(recurrent,non recurrent),
    shuffle each set, choose part of each set according the test/train ratio
    then combine the partial sets into train set and test set
    """
    def __init__(self, examples_list):
        self.recurrent_examples = self._get_recurrent_examples(examples_list)
        self.non_recurrent_examples = self._get_non_recurrent_examples(examples_list)


    def _get_recurrent_examples(self,examples_list):
        return list(filter(lambda example: example[0] > 0, examples_list))
 
    def _get_non_recurrent_examples(self,examples_list): 
        return list(filter(lambda example: example[0] < 0, examples_list)) 


    def get_train_and_test_data_by_ratio(self,ratio):
        size_of_recurrent_train_examples = int(len(self.recurrent_examples)*ratio)
        size_of_non_recurrent_train_examples = int(len(self.non_recurrent_examples)*ratio)
        
        size_of_recurrent_test_examples = int(len(self.recurrent_examples)) - size_of_recurrent_train_examples
        size_of_non_recurrent_test_examples = int(len(self.non_recurrent_examples)) - size_of_non_recurrent_train_examples

        shuffled_recurrent_examples = random.sample(self.recurrent_examples, len(self.recurrent_examples))
        shuffled_non_recurrent_examples = random.sample(self.non_recurrent_examples, len(self.non_recurrent_examples))

        train_data = shuffled_recurrent_examples[0:size_of_recurrent_train_examples] + shuffled_non_recurrent_examples[0:size_of_non_recurrent_train_examples]  
        test_data = shuffled_recurrent_examples[size_of_recurrent_train_examples:] + shuffled_non_recurrent_examples[size_of_non_recurrent_train_examples:]
        return train_data, test_data


    def extract_label_from_data(self,data):
        label_list = [list[0] for list in data]
        return numpy.array(label_list,dtype=numpy.double)

    def extract_features_from_labeled_data(self,data):
        feature_list = [list[1:] for list in data]
        return numpy.array(feature_list,dtype=numpy.double).astype(numpy.double);