import sys
import logging
import joblib
from yaml import safe_load
from pathlib import Path
import pandas as pd
from MLOps.logger import CustomLogger

from sklearn.ensemble import RandomForestClassifier

TARGET = 'case_status'

# custom logger for module
logger = CustomLogger('train')
# create a stream handler
console_handler = logging.StreamHandler()
# add console handler to the logger
logger.logger.addHandler(console_handler)

# read the dataframe
def read_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.log_message(f'{data_path.name} does not exist')
    else:
        return df
# make X and y
def make_X_and_y(df: pd.DataFrame,target_column) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_column)
    y = df[target_column]
    logger.log_message('Data split into X and y')
    return X, y

def read_parameters(params_file_path: str) -> dict:
    try:
        with open(params_file_path,'r') as params:
            parameters = safe_load(params)
    except Exception as e:
        logger.log_message('Failed to read params.yaml file')
    else:
        logger.log_message('Parameters read successfully from params file')
        return parameters

def fit_model(model, params, X_train, y_train):
    clf = model().set_params(**params)
    logger.log_message(f"Model fit with parameters = {clf.get_params()}")
    # fit the model
    clf.fit(X_train,y_train)
    return clf

def save_model(obj, save_path: Path):
    try:
        joblib.dump(value=obj,filename=save_path)
    except Exception as e:
        logger.log_message("Model save path does not exist")
    else:
        logger.log_message("Model saved to save path")



def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed"
    save_model_path = root_path / "models" / "classifiers"
    save_model_path.mkdir(exist_ok=True)
    logger.log_message("Classifiers folder created")
    metrics_save_location = root_path / "reports"
    metrics_save_location.mkdir(exist_ok=True)
    logger.log_message("Report folder created")

    # read the train filename from the cmd
    file_name = sys.argv[1]
    logger.log_message(f"Filename read from cmd is {file_name}")
    # read the training data
    train_df = read_data(data_path=data_path / file_name)
    # split the data into X and y
    X_train, y_train = make_X_and_y(df=train_df,target_column=TARGET)
    logger.log_message("Data split into X and y")
    model_params = read_parameters(params_file_path="params.yaml")['train']['RandomForestClassifier']
    # fit the model on training data
    clf = fit_model(RandomForestClassifier, 
                    model_params, 
                    X_train= X_train, y_train= y_train)
    logger.log_message("Model trained on training data")
    # save the model
    save_model(obj=clf,save_path=save_model_path / 'rf_clasfy.joblib')
    
    
if __name__ == '__main__':
    main()
    