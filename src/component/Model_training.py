import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def train(self, train_array, test_array):
        try:
            logging.info('Split Training and test input data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            logging.info('Training the model is started')
            param = { 'estimator__C': [0.1, 1, 10], 
                     'estimator__epsilon': [0.1, 0.2, 0.5], 
                     'estimator__kernel': ['linear', 'poly', 'rbf'], 
                     'estimator__gamma': ['scale', 'auto'] }
            
            bag = BaggingRegressor(estimator=SVR(), n_estimators=50, random_state=14)
            # Perform Grid Search
            grid_search = GridSearchCV(bag, param, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)

            # Get the best BaggingRegressor model with optimized hyperparameters
            best_bag = grid_search.best_estimator_

            # Evaluate the best model using cross-validation
            score = cross_val_score(best_bag, X_train, y_train, cv=5, scoring='r2')
            mean_score = np.mean(score)

            print("Best parameters:", grid_search.best_params_)
            print("Mean cross-validation R2 score:", mean_score)
            
            logging.info('Make predictions and evaluate the model')
            predicted = best_bag.predict(X_test)
            r_score = r2_score(y_test, predicted)
            
            logging.info('Save the model trainer')
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_bag
            )
            return r_score
 
        except Exception as e:
            raise CustomException(e,sys)