import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import sys
from src.utils import save_object_preprocessor

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
    
    def get_transformation_obj(self):
        try:
            numerical_cols = ["writing_score", "reading_score"]
            ordinal_impute_col = ["parental_level_of_education"] 
            nominal_impute_col = ["race_ethnicity"]
            nominal_mandatory_cols = ["gender", "lunch", "test_preparation_course"]


            logging.info(f'data transformation initiated')

        
            num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  
            ('scaler', StandardScaler())
                ])

            education_order = [
             ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
                ]
            ord_impute_pipeline = Pipeline(steps=[
            
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('encoder', OrdinalEncoder(categories=education_order))
            ])

        
            nom_impute_pipeline = Pipeline(steps=[
            
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))
            ])
        
            nom_mandatory_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipe', num_pipeline, numerical_cols),
                    ('ord_impute', ord_impute_pipeline, ordinal_impute_col), 
                    ('nom_impute', nom_impute_pipeline, nominal_impute_col),
                    ('nom_mandatory', nom_mandatory_encoder, nominal_mandatory_cols) 
             ],
                verbose_feature_names_out=False
                )
            logging.info(f'Preprocessor object created and is ready to be returned.')
            return preprocessor
            
        except Exception as e:
           raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data,test_data):

        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)

            target_column='math_score'

            training_input_fetaures=train_df.drop(columns=target_column,axis=1)
            training_target_feature=train_df[target_column]

            testing_input_fetaures=test_df.drop(columns=target_column,axis=1)
            testing_target_feature=test_df[target_column]

            logging.info(f"applying preprocessing")
            
            preprocessing_obj=self.get_transformation_obj()

            transformed_train_data=preprocessing_obj.fit_transform(training_input_fetaures)
            transformed_test_data=preprocessing_obj.transform(testing_input_fetaures)

            train_data_arr=np.c_[transformed_train_data,np.array(training_target_feature)]

            test_data_arr=np.c_[transformed_test_data,np.array(testing_target_feature)]

            logging.info(f"preprocessing object saved")
             
            #saveobject function is in utils.py
            save_object_preprocessor(file_path=self.data_transformation.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            return (
                train_data_arr,
                test_data_arr,
                self.data_transformation.preprocessor_obj_file_path)
        except Exception as e:
           raise CustomException(e,sys)