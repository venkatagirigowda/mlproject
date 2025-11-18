import os 
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor  
from sklearn.metrics import r2_score,mean_squared_error

from src.utils import save_object_model,eval_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('splitting training and test input data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'DecisionTree':DecisionTreeRegressor(),
                'RandomForest':RandomForestRegressor(),
                'GradientBoosting':GradientBoostingRegressor(), 
                'XGBRegressor':XGBRegressor(),
                'CatBoostingRegressor':CatBoostRegressor(verbose=False)
            }

            params = {
                    'LinearRegression': {},  # No tuning needed for standard Linear Regression
    
                    'DecisionTree': {
                        # Controls model complexity and prevents overfitting
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                    },
                    
                    'RandomForest': {
                        # n_estimators: Number of trees in the forest
                        'n_estimators': [100, 200, 400],
                        'max_depth': [10, 20, 30, None],
                        'min_samples_split': [5, 10],
                        'min_samples_leaf': [2, 4],
                    },
                    
                    'GradientBoosting': {
                        # n_estimators: Number of boosting stages
                        'n_estimators': [50, 100, 200],
                        # learning_rate: Controls contribution of each tree
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5],
                    },
                    
                    'XGBRegressor': {
                        # Recommended to start with smaller n_estimators and tune learning_rate
                        'n_estimators': [100, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        # subsample/colsample_bytree control variance
                        'subsample': [0.7, 1.0], 
                        'colsample_bytree': [0.7, 1.0],
                    },
                    
                    'CatBoostingRegressor': {
                        # iterations: Similar to n_estimators
                        'iterations': [50, 100, 200],
                        'learning_rate': [0.03, 0.1],
                        'depth': [4, 6, 8],
                        # l2_leaf_reg: Regularization parameter
                        'l2_leaf_reg': [1, 3, 5]
                    }
                }

            train_scores_report, test_scores_report = eval_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)


            best_model_name = max(test_scores_report, key=test_scores_report.get)


            best_model_test_score = test_scores_report[best_model_name]
            best_model_train_score = train_scores_report[best_model_name] 


            best_model = models[best_model_name]

            if best_model_test_score < 0.6:
             raise CustomException("No best model found or best model score is below minimum threshold (0.6)")

            logging.info(f"Best found model on testing dataset: {best_model_name} with Test R2 Score: {best_model_test_score:.4f}")
            logging.info(f"Diagnostic: Best model's Train R2 Score: {best_model_train_score:.4f}")

            # Save the best model object to disk
            save_object_model(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
                )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)