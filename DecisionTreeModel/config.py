class Config:
    def __init__(self):
        self.train_x_path = 'trainx.pkl'
        self.train_y_path = 'trainy.pkl'
        self.test_x_path = 'testx.pkl'
        self.test_y_path = 'testy.pkl'
        self.decision_tree_path = 'decision_tree.model'
        self.data_path = 'data9.csv'
        self.train_data_ratio = 0.8

    @property
    def order_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'order_201611{}'
        files = [file_pattern.format(date, date) for date in dates]
        return files

    @property
    def gps_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'gps_201611{}'
        files = [file_pattern.format(date, date) for date in dates]
        return files

    @property
    def order2vector_data_files(self):
        dates = [str(x).zfill(2) for x in list(range(1, 10))]
        file_pattern = 'order2vector_201611{}.csv'
        files = [file_pattern.format(date, date) for date in dates]
        return files

config = Config()
