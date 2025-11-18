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
        
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 writing_score: float,
                 reading_score: float):
        self.gender = gender,
        self.race_ethnicity = race_ethnicity,
        self.parental_level_of_education = parental_level_of_education,
        self.lunch = lunch, 
        self.test_preparation_course = test_preparation_course,
        self.writing_score = writing_score,
        self.reading_score = reading_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            raise CustomException(e, sys)
