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
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
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
            
            logging.info("Obtaining transformer object")
            transformer= self.get_transformer()
            
            logging.info("Applying transformer object on training dataframe and testing dataframe.")
            train_arr=transformer.fit_transform(train_df)
            test_arr=transformer.transform(test_df)
            
            logging.info(f"Saved preprocessing object.")
            save_object(
                 file_path=self.data_transformation_config.transformer_obj_file_path,
                 obj=transformer
                 )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.transformer_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)