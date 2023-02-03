# Description of DecisionTreeModel Folder

This folder contains the code of generating Decision Tree Model in Section 3.3.1 in paper Cross Online Ride-sharing for Shared Mobilities in Spatial Crowdsourcing. The experimental environment is based on python3.9.

## Description of **"config.py"**

 **config.py** contains the basic settings of the algorithms in this folder. All the urls can be modified according to personal preference.

## Description of "GenerateData.py"

**GenerateData.py** is used to generate the training datasets and test datasets. 

Since our goal is to predict the direction of a worker's next destination based on his/er driving status, we generate four features for the decision tree model: the start time of order, the start longitude, the start latitude, and the current driving angle. The lable of the decision tree model is the angle between the current driving direction and destination direction.

The input of GenerateData is the original gps data and order data. Since the original datasets are too large to be uploaded to GitHub, we provide an additional link for users to download the original datasets. You can choose different datasets and the number of datasets to generate different overall data, and the output path is “config.data_path” in config. You can change “config.train_data_ratio” in config to adjust the ratio of training data and test data, and the output paths are set in config.py.

Data Link: https://pan.baidu.com/s/1KaxU4g23Pdup14Auf1hzpg Password: 0bbw.



## Description of "DecisionTree.py"

**DecisionTree.py** is used to generate the decision tree model based on the training data and test data generated in **GenerateData.py**.

We use the method in sklearn.tree and the generated datasets to train the decision tree model. We export the trained model to “config.decision_tree_path” in config.
