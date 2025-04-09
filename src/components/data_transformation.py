import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config  = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ])
            cat_pipeline = Pipeline(
                steps=[
                    ('encoder',OneHotEncoder()),
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                  
                     
                ])
            
            logging.info('Numerical features Standard Scaler Completed')
            logging.info('Categorical features One Hot Encoding Completed')

            preprocessor = ColumnTransformer(
                [
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test dataset ')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformation_obj()
            target_column = 'math score'

            input_features_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns = [target_column],axis = 1)
            target_feature_test_df = test_df[target_column]
            
            logging.info('applying preprocessing object to thye train and test df')

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Preprocessing objectsv saved") 
        
            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )
        
            return (
                train_arr,
                test_arr,
                
                )
    
        except Exception as e:
            raise CustomException(e,sys)
