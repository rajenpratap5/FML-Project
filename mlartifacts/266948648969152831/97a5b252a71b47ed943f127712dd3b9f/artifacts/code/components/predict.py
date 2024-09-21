import sys
import logging
import joblib
import json
from pathlib import Path
import pandas as pd
from MLOps.logger import CustomLogger
import mlflow
from mlflow.sklearn import log_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TARGET = 'case_status'
MODEL_NAME = 'rf_clasfy.joblib'

# custom logger for module
logger = CustomLogger('train')
# create a stream handler
console_handler = logging.StreamHandler()
# add console handler to the logger
logger.logger.addHandler(console_handler)

# set the tracking uri
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# set the experiment name
mlflow.set_experiment("dvc pipeline")

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

def do_predictions(model,X):
    y_pred = model.predict(X)
    return y_pred

def load_model(model_path: Path):
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError as e:
        logger.log_message("Model Path does not exist")
    else:
        logger.log_message("Model Loaded from path successfully")
        return clf
    
def calculate_metrics(data_name: str,y,y_pred):
    metrics_dict = {}
    data = data_name.split("_")[0]
    
    metrics_dict[f'{data}_accuracy'] = accuracy_score(y,y_pred)
    metrics_dict[f'{data}_f1_score'] = f1_score(y,y_pred)
    metrics_dict[f'{data}_precision'] = precision_score(y,y_pred)
    metrics_dict[f'{data}_recall'] = recall_score(y,y_pred)
    
    return metrics_dict


def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed"    
    model_load_path = root_path / "models" / "classifiers"

    # load the model
    clf = load_model(model_path=model_load_path / MODEL_NAME)

    for ind in range(1,3):
        filename = sys.argv[ind]
        # name of data
        data = filename.split("_")[0]
        # read the data
        df = read_data(data_path=data_path / filename)
        # split into x and y
        X,y = make_X_and_y(df=df,target_column=TARGET)
        # get predictions
        y_pred = do_predictions(model=clf,X=X)

        if filename == "train_final.csv":
            # generate metrics
            train_metrics_dict = calculate_metrics(data_name=(data_path / filename).name,
                                            y=y,y_pred=y_pred)
        
            metrics_save_path = root_path / "reports" / f"{data}_metrics.json"
            # convert metrics dict to json
            with open(metrics_save_path,"w") as f:
                json.dump(train_metrics_dict,f,indent=4)

        elif filename == "test_final.csv":
             # generate metrics
            test_metrics_dict = calculate_metrics(data_name=(data_path / filename).name,
                                            y=y,y_pred=y_pred)
        
            metrics_save_path = root_path / "reports" / f"{data}_metrics.json"
            # convert metrics dict to json
            with open(metrics_save_path,"w") as f:
                json.dump(test_metrics_dict,f,indent=4)
            
        logger.log_message("Metrics saved to save location")
    
    logger.log_message("MlFlow logging started")
    # start mlflow tracking
    with mlflow.start_run() as run:
        # track the params
        mlflow.log_params(clf.get_params())
        logger.log_message("parameters logged")
        
        # track the train metrics
        mlflow.log_metrics(train_metrics_dict)
        logger.log_message("metrics logged")
        
        
        # track the test metrics
        mlflow.log_metrics(test_metrics_dict)
        logger.log_message("metrics logged")
        
        # log the model
        log_model(sk_model=clf,artifact_path="model")
        logger.log_message("model logged")
        
        # log the code
        mlflow.log_artifacts("MLOps","code")
        logger.log_message("MLOps folder logged")

    # register the model
    run_id = run.info.run_id
    artifact_path = "model"
    
    register_model_uri = f"runs:/{run_id}/{artifact_path}"
    
    mlflow.register_model(model_uri=register_model_uri,
                          name="Random_Forest_USvisa_Prediction")
    
if __name__ == "__main__":
    main()

