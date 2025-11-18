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

# --- IMPORTANT: Adds the parent directory to sys.path for internal imports ---
# This ensures that when the server runs from a subdirectory (like 'app' or 'api'), 
# it can find modules in the 'src' directory at the project root level.
# This should point to your project root where the 'src' folder is located.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Importing the necessary classes from your existing pipeline structure
# NOTE: The actual implementation of CustomData and PredictPipeline must exist
# in src.pipeline.predict_pipeline for this code to run successfully.
try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
except ImportError:
    # Placeholder classes if the actual pipeline is not present for testing
    class CustomData:
        def __init__(self, **kwargs): 
            # Ensure None/empty strings values from form become NaN for pipeline imputation
            self.data = {}
            for k, v in kwargs.items():
                # Convert None or empty string (from optional form field) to np.nan
                if v == '' or v is None:
                    self.data[k] = np.nan
                else:
                    self.data[k] = v
            
        def get_data_as_data_frame(self): 
            # Reorder columns to match the expected order of the ML pipeline
            feature_order = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
            ]
            
            # Create a DataFrame ensuring the columns are in the correct order
            return pd.DataFrame([self.data], columns=feature_order)


    class PredictPipeline:
        def predict(self, df):
            # Mock prediction result for demonstration
            print("--- Mock Prediction Run ---")
            print("Input Data received by Pipeline:")
            print(df)
            return [75.5] 

    print("WARNING: CustomData and PredictPipeline imports failed. Using mock classes.")


app = FastAPI(title="Student Performance Prediction API")

# Setup Jinja2 templating for rendering HTML forms
# The directory is set to "." so it looks for index.html in the same folder.
templates = Jinja2Templates(directory=".") 

# --- Pydantic Model for API Request Validation ---
class PredictionInput(BaseModel):
    # Mandatory fields (App must enforce these)
    gender: str
    ethnicity: str = None # Optional in form, passed as str or None
    lunch: str
    test_preparation_course: str
    writing_score: float
    reading_score: float
    
    # Optional field
    parental_level_of_education: str = None

    # Custom dependency to parse form data into the Pydantic model
    @classmethod
    def as_form(cls, 
                gender: str = Form(...), 
                ethnicity: str = Form(None), # Accepts None/blank string for optional
                parental_level_of_education: str = Form(None), # Accepts None/blank string for optional
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
    # Renamed template to 'index.html'
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request, 
    data: PredictionInput = Depends(PredictionInput.as_form)
):
    """Route for handling the form submission and prediction."""
    try:
        # 1. Map Pydantic data to CustomData object
        custom_data_object = CustomData(
            gender=data.gender,
            race_ethnicity=data.ethnicity, # Note: Mapping form field 'ethnicity' to model field 'race_ethnicity'
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=data.reading_score,
            writing_score=data.writing_score
        )

        # Convert to DataFrame
        pred_df = custom_data_object.get_data_as_data_frame()
        print(f"Prediction Dataframe:\n{pred_df}")

        # 2. Run Prediction Pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # 3. Render results back to the home page
        final_result = f"{results[0]:.2f}"
        
        # Renamed template to 'index.html'
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": final_result}
        )

    except Exception as e:
        # Log the full error traceback for debugging
        import traceback
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        
        # Renamed template to 'index.html'
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": f"Prediction Failed: Check console for trace. Error: {e}"}
        )

# Optional: Run with uvicorn if script is executed directly
if __name__ == "__main__":
    # HOST set to "127.0.0.1" (localhost only)
    uvicorn.run(app, host="127.0.0.1", port=8000)