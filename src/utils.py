import os
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import customException
import sys


def save_object_preprocessor(file_path,obj):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file:
        dill.dump(obj,file)


def save_object_model(file_path,obj):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file:
        pickle.dump(obj,file)

        

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import logging

def eval_models(X_train, y_train, X_test, y_test, models, params):
    
    train_scores = {}
    test_scores = {}
    
    # 1. Use .items() for clean iteration
    for model_name, model_obj in models.items():
        
        # Access parameters correctly using the model_name (Fix to Issue 1)
        param_grid = params.get(model_name)
        
        # Skip tuning if no parameters are provided (e.g., Linear Regression)
        if not param_grid:
            logging.info(f"Fitting base model for {model_name} (no tuning).")
            final_model = model_obj
            final_model.fit(X_train, y_train)
        
        else:
            logging.info(f"Starting GridSearchCV for {model_name}...")
            
            # 2. Perform Grid Search
            gs = GridSearchCV(
                model_obj, 
                param_grid, 
                cv=3, 
                scoring='r2', 
                verbose=0, 
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            
        
            final_model = gs.best_estimator_
            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
        # 4. Predict and Score using the Final Model
        
        y_train_pred = final_model.predict(X_train)
        y_test_pred = final_model.predict(X_test)

        train_scores[model_name] = r2_score(y_train, y_train_pred)
        test_scores[model_name] = r2_score(y_test, y_test_pred)
            
    return train_scores, test_scores

def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            obj=dill.load(file)
        return obj
    except Exception as e:
        raise customException(e,sys)