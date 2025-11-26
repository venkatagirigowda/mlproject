import os
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            
            logging.info('Loading preprocessor and model objects')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scale=preprocessor.transform(features)
            preds=model.predict(data_scale)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
