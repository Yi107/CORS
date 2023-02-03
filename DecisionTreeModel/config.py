class Config:
    def __init__(self):
        #The url of generated data
        self.data_path = 'data.csv'
        #The ratio of the number of training data to the number of test data
        self.train_data_ratio = 0.8
        #The url of training data and test data
        self.train_x_path = 'trainx.pkl'
        self.train_y_path = 'trainy.pkl'
        self.test_x_path = 'testx.pkl'
        self.test_y_path = 'testy.pkl'
        #The url of generated decision tree model
        self.decision_tree_path = 'decision_tree.model'
        
        
    #The url of the original order data
    @property
    def order_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'order_201611{}'
        files = [file_pattern.format(date, date) for date in dates]
        return files
    
    #The url of the original gps data
    @property
    def gps_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'gps_201611{}'
        files = [file_pattern.format(date, date) for date in dates]
        return files
    #The url of the start vector of order data which is used in training data
    @property
    def order2vector_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'order2vector_201611{}.csv'
        files = [file_pattern.format(date, date) for date in dates]
        return files

config = Config()
