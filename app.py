from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import uvicorn
from data_model import PredictionDataset
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from MLOps.logger import CustomLogger
import logging

# custom logger for module
logger = CustomLogger('train')
# create a stream handler
console_handler = logging.StreamHandler()
# add console handler to the logger
logger.logger.addHandler(console_handler)


app = FastAPI()

current_file_path = Path(__file__).parent

model_path = current_file_path / "models" / "classifiers" / "rf_clasfy.joblib"
preprocessor_path = model_path.parent.parent / "transformer" / "col_transformer.joblib"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ('classify',model)
])

# Converting the year of establishment to age of company
def calculate_company_age(df: pd.DataFrame, year_column: str = 'yr_of_estab') -> pd.DataFrame:
    current_year = datetime.now().year
    df['company_age'] = current_year - df[year_column]
    return df

# Dropping the column
def drop_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.drop(columns=[column_name], axis=1)

@app.get('/')
def home():
    return "Welcome to USA Visa prediction app"

@app.post('/predictions')
def do_predictions(test_data:PredictionDataset):
    logger.log_message(f"Received prediction request: {test_data}")
    logger.log_message("Data conversion started")
    X_test = pd.DataFrame(
        data = {
            "case_id": test_data.case_id,
            "continent": test_data.continent,
            "education_of_employee": test_data.education_of_employee,
            "has_job_experience": test_data.has_job_experience,
            "requires_job_training": test_data.requires_job_training,
            "no_of_employees": test_data.no_of_employees,
            "yr_of_estab": test_data.yr_of_estab,
            "region_of_employment": test_data.region_of_employment,
            "prevailing_wage": test_data.prevailing_wage,
            "unit_of_wage": test_data.unit_of_wage,
            "full_time_position": test_data.full_time_position
         }, index=[0]
    )
    logger.log_message("X_test done")
    
    # Calculating company age
    X_test = calculate_company_age(X_test)
    logger.log_message("company age calculated")
    # dropping column year of establishment and case_id
    X_test = drop_column(X_test, "case_id")
    X_test = drop_column(X_test, "yr_of_estab")
    
    prediction = model_pipe.predict(X_test)
    return f"Prediction: {prediction}"


if __name__ == "__main__":
    uvicorn.run(app="app:app",
                host="0.0.0.0",
                port=8000, reload=True)