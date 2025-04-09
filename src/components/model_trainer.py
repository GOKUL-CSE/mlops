import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    model_trained_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    
    def __init__(self):
    
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Train Test splitting started")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                 "Linear Regression": LinearRegression(),
                 "K Nearest Neighbour":KNeighborsRegressor(),
                 "SVR":SVR(),
                 "Decision Tree":DecisionTreeRegressor(),
                 "RandomForest" :RandomForestRegressor(),
                 "Adaboost":AdaBoostRegressor(),
                 "Gradient Boosting":GradientBoostingRegressor(),
                 "CatBoost" :CatBoostRegressor(),
                 "XGB ":XGBRegressor(),

                }
            model_report = evaluate_model(X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_test,
                                          y_test=y_test,
                                          models=models)
            
           
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No good model found")
            logging.info("Best  model found on both train and test data" )

            save_object(
                file_path= self.model_trainer_config.model_trained_file_path,
                obj=best_model_name
            )
            logging.info("model saved as pickle file")

            prediction = best_model.predict(X_test)
            logging.info("Prediction done")

            r2score = r2_score(y_test,prediction)

            return r2score

        except Exception as e:
            raise CustomException(e,sys)
            