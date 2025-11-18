import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
         logging.info("Entered the data ingestion method or component")
         try:
             df=pd.read_csv('notebook/data/stud.csv')
             logging.info("Read the dataset as dataframe")

             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

             df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

             logging.info('train_test splits intiated ')

             train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True )
             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

             logging.info('Ingestion of the data is completed')

             return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
             )
         except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    data_ingestion_obj=DataIngestion()
    train_data,test_data=data_ingestion_obj.initiate_data_ingestion()
    
    transformation_obj=DataTransformation()
    train_data_arr,test_data_arr,_=transformation_obj.initiate_data_transformation(train_data,test_data)

    model_obj=ModelTrainer()
    print(model_obj.initiate_model_training(train_data_arr,test_data_arr))
    

             
    