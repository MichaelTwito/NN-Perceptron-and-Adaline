import pandas


class parser(object):
    """
    This class will parse the dataset into pandas dataframe,
    then replace the R and N of the data into 1 and -1,
    then rerutrn a dictionary where id's are the keys, 
    and the examples will be the corresponded arrays
    """

    def __init__(self, dataset_filename):
        self.data_frame = pandas.read_csv(dataset_filename, header=None)
        self._normalized_dataset = self._normalize_dataset()

    def __call__(self):
        return self._normalized_dataset

    def _normalize_dataset(self):
        self.data_frame[34] = self.data_frame[34].replace(['?'],3)
        self.data_frame[1] = self.data_frame[1].replace(['R'], 1)
        self.data_frame[1] = self.data_frame[1].replace(['N'], -1)
        # self.data_frame[self.data_frame.shape[1] - 1] = self.data_frame[self.data_frame.shape[1] - 1].replace(['?'], 0)
        self.data_frame[34] = self.data_frame[34].astype(int)

        return self.data_frame.set_index(0).T.to_dict('list')
