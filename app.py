from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
except ImportError:

    class CustomData:
        def __init__(self, **kwargs): 
            self.data = {}
            for k, v in kwargs.items():
                if v == '' or v is None:
                    self.data[k] = np.nan
                else:
                    self.data[k] = v
            
        def get_data_as_data_frame(self): 
            feature_order = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
            ]
            
            return pd.DataFrame([self.data], columns=feature_order)


    class PredictPipeline:
        def predict(self, df):
            print("--- Mock Prediction Run ---")
            print("Input Data received by Pipeline:")
            print(df)
            return [75.5] 

    print("WARNING: CustomData and PredictPipeline imports failed. Using mock classes.")


app = FastAPI(title="Student Performance Prediction API")

templates = Jinja2Templates(directory=".") 


class PredictionInput(BaseModel):
    gender: str
    ethnicity: str = None 
    lunch: str
    test_preparation_course: str
    writing_score: float
    reading_score: float
    
    parental_level_of_education: str = None

    @classmethod
    def as_form(cls, 
                gender: str = Form(...), 
                ethnicity: str = Form(None),
                parental_level_of_education: str = Form(None), 
                lunch: str = Form(...),
                test_preparation_course: str = Form(...),
                writing_score: float = Form(...),
                reading_score: float = Form(...)):
        return cls(
            gender=gender,
            ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            writing_score=writing_score,
            reading_score=reading_score
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Route for the home page."""
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request, 
    data: PredictionInput = Depends(PredictionInput.as_form)
):
    """Route for handling the form submission and prediction."""
    try:
        custom_data_object = CustomData(
            gender=data.gender,
            race_ethnicity=data.ethnicity, 
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=data.reading_score,
            writing_score=data.writing_score
        )

    
        pred_df = custom_data_object.get_data_as_data_frame()
        print(f"Prediction Dataframe:\n{pred_df}")

      
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

      
        final_result = f"{results[0]:.2f}"
        
        
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": final_result}
        )

    except Exception as e:
        
        import traceback
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        
        
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": f"Prediction Failed: Check console for trace. Error: {e}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)