from Utils.DataSetParser import parser
from Utils.DataSetHandler import handler


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
    
    train_dataset, test_dataset = plain_dataset.get_train_and_test_data_by_ratio(0.66)

    print (len(train_dataset))

    print (len(test_dataset))



if __name__ == "__main__":
    main()