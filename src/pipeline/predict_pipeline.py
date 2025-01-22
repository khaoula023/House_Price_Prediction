import sys
import pandas as pd
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            logging.info("Loading artifacts")
            model_path = os.path.abspath('artifacts/model.pkl')
            transformer_path = os.path.abspath('artifacts/transformer.pkl')
            target_transformer_path = os.path.abspath('artifacts/target_transformer.pkl')
            model = load_object(file_path=model_path)
            transformer = load_object(file_path=transformer_path)
            target_transformer = load_object(file_path=target_transformer_path)
            
            logging.info('Data transformation')
            data_scaled = transformer.transform(features)
            
            logging.info('Make Predictions')
            predictions = model.predict(data_scaled).reshape(-1, 1)
            predictions = target_transformer.inverse_transform(predictions).flatten()
            return predictions
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, Area: int, Bedrooms: int, Bathrooms: int, Floors: int, YearBuilt: int, Location: str, Condition: str, Garage: str):
        self.Area = Area
        self.Bedrooms = Bedrooms
        self.Bathrooms = Bathrooms
        self.Floors = Floors
        self.YearBuilt = YearBuilt
        self.Location = Location
        self.Condition = Condition
        self.Garage = Garage
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Area": [self.Area],
                "Bedrooms": [self.Bedrooms],
                "Bathrooms": [self.Bathrooms],
                "Floors": [self.Floors],
                "YearBuilt": [self.YearBuilt],
                "Location": [self.Location],
                "Condition": [self.Condition],
                "Garage": [self.Garage]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    data = CustomData(4076, 5, 4, 2, 1985, 'Rural', 'Excellent', 'No')
    features = data.get_data_as_data_frame()
    obj = PredictPipeline()
    prediction = obj.predict(features)
    print(f'The price is: {prediction}')
