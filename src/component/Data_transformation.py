import os 
import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    transformer_obj_file_path = os.path.join('artifacts', 'transformer.pkl')
    target_transformer_obj_file_path = os.path.join('artifacts', 'target_transformer.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_transformer(self):
        logging.info('Entered the data transformation method or component')
        try:
            numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
            categorical_features = ['Location', 'Condition', 'Garage']
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info("numerical columns encoding completed")
            logging.info("categorical columns encoding completed")
            
            transformer = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            return transformer
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform(self, train_path, test_path):
        try:
            logging.info('Read train and test :')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('remove ID column')
            train_df = train_df.drop(columns=['Id'], axis=1)
            test_df = test_df.drop(columns=['Id'], axis=1)
            
            logging.info("Obtaining transformer object")
            transformer= self.get_transformer()
            
            logging.info('Preparing the features and the target columns')
            target = 'Price'
            features_train_df = train_df.drop(columns=[target],axis=1)
            target_train_df=train_df[[target]]
            
            features_test_df=test_df.drop(columns=[target],axis=1)
            target_test_df=test_df[[target]]
            
            logging.info("Applying transformer object on training dataframe and testing dataframe.")
            features_train_arr=transformer.fit_transform(features_train_df)
            features_test_arr=transformer.transform(features_test_df)
            
            logging.info('Normalizing the target column')
            target_transformer = StandardScaler()  
            target_train_arr = target_transformer.fit_transform(target_train_df)
            target_test_arr = target_transformer.transform(target_test_df)
            
            logging.info('Concatenate the results')
            train_arr = np.c_[features_train_arr, target_train_arr]
            test_arr = np.c_[features_test_arr, target_test_arr]
            logging.info(f"Saved transformer objects.")
            save_object(file_path=self.data_transformation_config.transformer_obj_file_path, obj=transformer)
            save_object(file_path=self.data_transformation_config.target_transformer_obj_file_path, obj=target_transformer)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.transformer_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)