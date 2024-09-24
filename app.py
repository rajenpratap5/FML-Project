from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
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

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None
        

    async def get_usvisa_data(self):
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")

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

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "usvisa.html",{"request": request, "context": "Rendering"})

@app.post("/")
async def do_predictions(request: Request):
    logger.log_message(f"Received prediction request")
    logger.log_message("Data conversion started")
    form = DataForm(request)
    await form.get_usvisa_data()
    
    form_data = {
                    'continent': form.continent,
                    'education_of_employee': form.education_of_employee,
                    'has_job_experience': form.has_job_experience,
                    'requires_job_training': form.requires_job_training,
                    'no_of_employees': form.no_of_employees,
                    'company_age': form.company_age,
                    'region_of_employment': form.region_of_employment,
                    'prevailing_wage': form.prevailing_wage,
                    'unit_of_wage': form.unit_of_wage,
                    'full_time_position': form.full_time_position,
    }
    logger.log_message("usvisa_data done")
    usvisa_data = pd.DataFrame(form_data, index=[0])
    logger.log_message("usvisa_data converted to dataframe")
    # Calculating company age
    # usvisa_data = calculate_company_age(usvisa_data)
    # logger.log_message("company age calculated")
    # dropping column year of establishment and case_id
    # usvisa_data = drop_column(usvisa_data, "case_id")
    # usvisa_data = drop_column(usvisa_data, "yr_of_estab")
    
    prediction = model_pipe.predict(usvisa_data)[0]

    status = None
    if prediction == 1:
        status = "Visa-approved"
    else:
        status = "Visa Not-Approved"

    return templates.TemplateResponse(
        "usvisa.html",
        {"request": request, "context": status},
    )
    




if __name__ == "__main__":
    uvicorn.run(app="app:app",
                host="0.0.0.0",
                port=8000, reload=True)