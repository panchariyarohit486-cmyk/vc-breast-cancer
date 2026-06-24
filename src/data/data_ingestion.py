# url --> changes ->>> local copy ->> train test split ->> save data
import numpy as np
import pandas as pd
import os
import sklearn.datasets
from sklearn.model_selection import train_test_split
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)['data_ingestion']

TEST_SIZE   = params['test_size']
RANDOM_STATE = params['random_state']
RAW_DATA_PATH = params['raw_data_path']

breast_canser_dataset = sklearn.datasets.load_breast_cancer()


#loading data to pandas data frame
data_frame = pd.DataFrame(breast_canser_dataset.data , columns = breast_canser_dataset.feature_names)

data_frame['label'] = breast_canser_dataset.target


x = data_frame.drop(columns='label',axis = 1)
y = data_frame['label']


x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size=TEST_SIZE,random_state=RANDOM_STATE)

data_path = os.path.join(RAW_DATA_PATH)
os.makedirs(data_path,exist_ok=True)
x_train.to_csv(os.path.join(data_path,'train.csv'),index=False)
x_test.to_csv(os.path.join(data_path,'test.csv'),index=False)
y_train.to_csv(os.path.join(data_path,'y_train.csv'),index=False)
y_test.to_csv(os.path.join(data_path,'y_test.csv'),index=False)

#stage 1--|--> -n name 
#         |--> -d src/data_ingestion.py
#         |--> -p params.yaml