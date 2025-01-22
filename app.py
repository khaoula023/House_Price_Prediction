from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import  CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('website.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else: 
        data = CustomData(
            Area=float(request.form.get('Area')),
            Bedrooms=int(request.form.get('Bedrooms')),
            Bathrooms = int(request.form.get('Bathrooms')),
            Floors=int(request.form.get('Floors')),
            YearBuilt=int(request.form.get('YearBuilt')),
            Location=request.form.get('Location'),
            Condition=request.form.get('Condition'),
            Garage=request.form.get('Garage')
            
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        result = round(result[0], 2)
        return render_template('website.html', results= result)
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug= True)