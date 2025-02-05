import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.component.Data_transformation import DataTransformation
from src.component.Model_training import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
         self.ingestion_config = DataIngestionConfig
    
    def get_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try: 
            df = pd.read_csv('notebook\data\House Price Prediction Dataset.csv')
            logging.info('Read the dataset as DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train Test split initiated')
            train, test = train_test_split(df,test_size=0.2, random_state=42)
            
            logging.info("Saving the Train and Test datasets")
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.get_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.transform(train_data, test_data)
    
    model = ModelTrainer()
    r2_score = model.train(train_arr, test_arr)
    print(f'R2 score = {r2_score}')