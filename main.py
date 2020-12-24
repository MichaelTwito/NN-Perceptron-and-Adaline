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

    number_of_successes = true_positive_precent + false_negative_precent

    if (true_positive_counter + true_negative_counter + false_positive_counter + false_negative_counter == len(test_label)):
        if (true_positive_counter + false_positive_counter !=  len(test_label) and true_negative_counter + false_negative_counter !=  len(test_label)):
            print ("True positive: {}".format(true_positive_counter) + " precent: {}%".format(true_positive_precent))
            print ("True negative: {}".format(true_negative_counter)+ " precent: {}%".format(true_negative_precent))
            print ("False positive: {}".format(false_positive_counter)+ " precent: {}%".format(false_positive_precent))
            print ("False negative: {}".format(false_negative_counter)+ " precent: {}%".format(false_negative_precent))
            print ("Success rate: {}%\n\n".format(true_positive_precent + false_negative_precent)) 
        else:
            number_of_successes = -1
    else: 
        number_of_successes = -1 
    
    return number_of_successes

def run_backpropagation_with_shuffled_data(number_of_runnings,plain_dataset,ratio=0.66,iterations=10000,hidden_layer_size=33):
    print ("Dataset ratio: {}% for train, the rest for test".format(ratio*100))
    total_runs_scoring = 0
    good_results_counter = 0
    for i in range(number_of_runnings):
        train_dataset, test_dataset = plain_dataset.get_train_and_test_data_by_ratio(ratio)
        train_unlabled = plain_dataset.extract_features_from_labeled_data(train_dataset)
        test_unlabled = plain_dataset.extract_features_from_labeled_data(test_dataset)
        train_label = plain_dataset.extract_label_from_data(train_dataset)
        test_label = plain_dataset.extract_label_from_data(test_dataset)
        number_of_successes = run_backpropagation(train_unlabled,train_label,test_unlabled,test_label,iterations, hidden_layer_size)  
        
        if (number_of_successes != -1):
            total_runs_scoring += number_of_successes
            good_results_counter += 1

    print ("Average succsess rate: {}%".format(total_runs_scoring/good_results_counter))

def omit_keys(dataset_dictionary):
    return list(map(lambda key:dataset_dictionary[key], dataset_dictionary.keys()))


def main():
    dataset_filename = 'Dataset/wpbc.data'
    normalized_dataset = parser(dataset_filename)

    dataset_as_list = omit_keys(normalized_dataset())
    
    plain_dataset = handler(dataset_as_list)
    
    ratio = 0.66
    
    train_dataset, test_dataset = plain_dataset.get_train_and_test_data_by_ratio(ratio)

    train_unlabled = plain_dataset.extract_features_from_labeled_data(train_dataset)
    test_unlabled = plain_dataset.extract_features_from_labeled_data(test_dataset)
    train_label = plain_dataset.extract_label_from_data(train_dataset)
    test_label = plain_dataset.extract_label_from_data(test_dataset)
    # perceptron = Perceptron(dim=len(train_unlabled[0]))
    # perceptron.train(data = train_unlabled,label=train_label,logs=True)
    numer_of_runnings = 50
    run_backpropagation_with_shuffled_data(numer_of_runnings,plain_dataset)

   

   
if __name__ == "__main__":
    main()