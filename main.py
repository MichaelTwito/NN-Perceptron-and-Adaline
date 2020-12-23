from Algorithms.Perceptron import Perceptron
from Algorithms.BackPropogation import BackPropogation
from Utils.DataSetParser import parser
from Utils.DataSetHandler import handler
from Utils.Score import Score
import numpy as np

def run_backpropagation(train_unlabled,train_label,test_unlabled,test_label,iterations=10000, hidden_layer_size=33):
    train_label = [[train_label] for train_label in train_label]
    
    backpropogation = BackPropogation(hidden_layer_size)

    for i in range(iterations):
        backpropogation.train(train_unlabled, train_label)
   
    
    result = backpropogation.feedForward(test_unlabled)
    result_flattered = np.concatenate(result).ravel().tolist()
    
    scoring = Score(result_flattered, test_label)

    true_positive_counter, true_negative_counter, false_positive_counter, false_negative_counter = scoring.get_score()
    true_positive_precent, true_negative_precent, false_positive_precent, false_negative_precent = scoring.get_normallize_score()

    print ("True positive: {}".format(true_positive_counter) + " precent: {}%".format(true_positive_precent))
    print ("True negative: {}".format(true_negative_counter)+ " precent: {}%".format(true_negative_precent))
    print ("False positive: {}".format(false_positive_counter)+ " precent: {}%".format(false_positive_precent))
    print ("False negative: {}".format(false_negative_counter)+ " precent: {}%".format(false_negative_precent))
    print ("Success rate: {}%\n\n".format(true_positive_precent + false_negative_precent))    

def omit_keys(dataset_dictionary):
    return list(map(lambda key:dataset_dictionary[key], dataset_dictionary.keys()))


def main():
    dataset_filename = 'Dataset/wpbc.data'
    normalized_dataset = parser(dataset_filename)
    
    # This omi_keys doesn't suppose to be here,
    # I want to put it inside the Parser
    # but got some shit errors with the object.

    dataset_as_list = omit_keys(normalized_dataset())
    
    plain_dataset = handler(dataset_as_list)
    
    ratio = 0.66
    print ("Dataset ratio: {}% for train, the rest for test".format(ratio*100))

    train_dataset, test_dataset = plain_dataset.get_train_and_test_data_by_ratio(ratio)

    train_unlabled = plain_dataset.extract_features_from_labeled_data(train_dataset)
    test_unlabled = plain_dataset.extract_features_from_labeled_data(test_dataset)
    train_label = plain_dataset.extract_label_from_data(train_dataset)
    test_label = plain_dataset.extract_label_from_data(test_dataset)
    # perceptron = Perceptron(dim=len(train_unlabled[0]))
    
    iterations = 10000
    hidden_layer_size = 25
    for j in range(1,10):

        print ("---------------------------------------------------------------------------------------------\n",
                "Iterations number: {}".format(iterations) + " Hidden layer size: {}\n".format(hidden_layer_size),
               "---------------------------------------------------------------------------------------------\n")

        for i in range(3):
            iterations = j * 10000 
            hidden_layer_size = j + 24
            run_backpropagation(train_unlabled,train_label,test_unlabled,test_label,iterations, hidden_layer_size)


   
if __name__ == "__main__":
    main()