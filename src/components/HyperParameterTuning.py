
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model
import os
import sys
class HyperParameterConifg:
    hyperparameters_file_path = os.path.join("artifacts","hyperparameters.pkl")

class HyperParameter:
    def __init__(self):
        self.HyperParameter_config = HyperParameterConifg()


    def initiate_hyperparametertuning(self ,X_train,y_train,X_test,y_test , models):
        try:


            params={
                "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse',   'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                    "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree',  'kd_tree', 'brute'],
                    'leaf_size': [20, 30, 40],
                    'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
                     }
                
            }

            model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test, models=models,param=params)
            save_object(
                file_path= self.HyperParameter_config.hyperparameters_file_path , 
                obj = model_report
            )
            return model_report
        except Exception as e:
            raise CustomException(e,sys)