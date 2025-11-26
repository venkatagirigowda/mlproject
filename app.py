from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,Field
import numpy as np
import pandas as pd
import uvicorn
from typing import Annotated
from src.pipeline.predict_pipeline import PredictPipeline

app=FastAPI()

class StudentData(BaseModel):
    gender: Annotated[str, Field(title="Gender", description="Enter the gender", examples=["male", "female"])]
    race_ethnicity: Annotated[str ,Field(title="Race/Ethinicity",description="Enter the group of ethnicity",examples=["group A","group B","group C","group D","group E"])]
    parental_level_of_education: Annotated[str, Field(title="Parental Level of Education", description="Enter the highest level of education attained by the parent(s)", examples=["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])]
    lunch: Annotated[str, Field(title="Lunch", description="Enter the type of lunch", examples=["standard", "free/reduced"])]
    test_preparation_course: Annotated[str, Field(title="Test Preparation Course", description="Enter the test preparation course status", examples=["none", "completed"])]
    reading_score: Annotated[float, Field(title="Reading Score", description="Enter the reading score", examples=[72])]
    writing_score: Annotated[float, Field(title="Writing Score", description="Enter the writing score", examples=[77])]

@app.get("/", response_class=HTMLResponse)
async def read_root(request : Request):
    templates = Jinja2Templates(directory=".")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  gender: Annotated[str, Form()],
                  race_ethnicity: Annotated[str, Form()],
                  parental_level_of_education: Annotated[str, Form()],
                  lunch: Annotated[str, Form()],
                  test_preparation_course: Annotated[str, Form()],
                  reading_score: Annotated[float, Form()],
                  writing_score: Annotated[float, Form()]):
    templates = Jinja2Templates(directory=".")

    try:
        input_data = StudentData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score)
        input_df = pd.DataFrame([input_data.model_dump()])
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(input_df)
        result = predict_pipeline.predict(input_df)
        predicted_score = result[0]
        predicted_score = max(0, min(100, predicted_score))
        predicted_score = round(predicted_score, 2)

        return templates.TemplateResponse("index.html", {"request": request, "predicted_score": predicted_score})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)