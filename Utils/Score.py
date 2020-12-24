class Score(object):
    """
    This object represents an Score printer.
    Receive an JSON object and return a nice looking table.
    """
    def __init__(self,predicted_list, actual_list):
        self.predicted_list = predicted_list
        self.actual_list = actual_list
        self.true_positive_counter = 0
        self.true_negative_counter = 0
        self.false_positive_counter = 0
        self.false_negative_counter = 0


    def get_score(self):
        
        list_length = len(self.actual_list)
        for i in range (list_length):
            if (self.actual_list[i] == 1 and self.predicted_list[i] == 1):
                self.true_positive_counter += 1
            elif (self.actual_list[i] == 1 and self.predicted_list[i] == -1):
                self.true_negative_counter += 1
            elif (self.actual_list[i] == -1 and self.predicted_list[i] == 1):
                self.false_positive_counter += 1
            elif (self.actual_list[i] == -1 and self.predicted_list[i] == -1):
                self.false_negative_counter += 1

        return self.true_positive_counter, self.true_negative_counter, self.false_positive_counter, self.false_negative_counter

    def get_normallize_score(self):
        list_length = len(self.actual_list)
        true_positive_precent = (self.true_positive_counter / list_length) * 100
        true_negative_precent = (self.true_negative_counter / list_length) * 100
        false_positive_precent = (self.false_positive_counter / list_length) * 100
        false_negative_precent =  (self.false_negative_counter / list_length) * 100
        
        return true_positive_precent, true_negative_precent, false_positive_precent, false_negative_precent