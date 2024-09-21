import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from MLOps.logger import CustomLogger
from yaml import safe_load

from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline

TARGET = 'case_status'
# Defining the custom logger
logger = CustomLogger(logger_name='feature_engineering')

#Create a console handler
console_handler = logging.StreamHandler()

# Add console handler to the logger
logger.logger.addHandler(console_handler)

# read the dataframe
def read_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


# Converting target column in ordinal encoding (Denied = 1, certified = 0)
def encode_target(target_column: pd.Series) -> pd.Series:
    target_column = np.where(target_column == 'Denied', 1, 0)
    return target_column

# Converting the year of establishment to age of company
def calculate_company_age(df: pd.DataFrame, year_column: str = 'yr_of_estab') -> pd.DataFrame:
    current_year = datetime.now().year
    df['company_age'] = current_year - df[year_column]
    return df

# Dropping the column
def drop_column(df: pd.DataFrame, column_name: str = 'yr_of_estab') -> pd.DataFrame:
    return df.drop(columns=[column_name])


# make X and y
def make_X_and_y(df: pd.DataFrame,target_column) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_column)
    y = df[target_column]
    logger.log_message('Data split into X and y')
    return X, y

def read_parameters(params_file_path: str) -> dict:
    with open(params_file_path,'r') as params:
        parameters = safe_load(params)['feature_engineering']
    logger.log_message('Parameters read successfully from params file')
    return parameters

def column_transformer(X_train: pd.DataFrame, col_to_transform: dict):
    preprocess = ColumnTransformer([('ordinalencoding', OrdinalEncoder(), col_to_transform['ord_col']),
                                ('onehotencoding', OneHotEncoder(), col_to_transform['oh_col']),
                                ('powertransform',PowerTransformer('yeo-johnson'),col_to_transform['pow_tran'])],
                                remainder='passthrough', verbose=True)
    preprocess.fit(X_train)
    return preprocess

def transform_data(X,transformer):
    X_trans = pd.DataFrame(transformer.transform(X))
    logger.log_message('Data tansformed through transformer')
    logger.log_message(f'Shape of data after transformation is {X_trans.shape}')
    return X_trans

def save_transformer(obj, save_path: Path):
    joblib.dump(value=obj,filename=save_path)

def save_the_data(dataframe: pd.DataFrame, data_path: Path) -> None:
    try:
        # save the data to path
        dataframe.to_csv(data_path,index=False)
        logger.log_message(f"DataFrame {data_path.name} saved successfuly")
    except Exception as e:
        logger.log_message(f'Dataframe not saved')


def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "raw"
    save_data_path = root_path / "data" / "processed"
    save_data_path.mkdir(exist_ok=True, parents=True)
    logger.log_message("Processed data folder created")
    save_transformer_path = root_path / "models" / "transformer"
    save_transformer_path.mkdir(exist_ok=True, parents=True)
    logger.log_message("Transformer model folder created")

    for ind in range(1,3):

        file_name = data_path / sys.argv[ind]
        # read the data
        df = read_data(data_path / file_name)
        logger.log_message(f'{file_name} read successfuly')

        # Calculating company age
        df = calculate_company_age(df)

        # dropping column year of establishment
        df = drop_column(df)

        feat_to_tansform = read_parameters('params.yaml')
        logger.log_message(f'The features to be transformed read from params file is {feat_to_tansform}')

        X, y = make_X_and_y(df, TARGET)

        if file_name.name == 'train.csv':
            #fitting the model 
            model = column_transformer(X, feat_to_tansform)
            # tranform the data
            X_trans = transform_data(X, model)
            # save transformer
            save_transformer(model,save_transformer_path / "col_transformer.joblib")
            # encoding the target column
            y_trans = encode_target(y)
            # concatenating the target column
            X_trans[TARGET] = y_trans
            logger.log_message(f'Shape of the final dataframe is {X_trans.shape}')
            # save the data
            save_the_data(X_trans,save_data_path / 'train_final.csv')

        elif file_name.name == "test.csv":
            # transform the data
            X_trans = transform_data(X,model)
            # transform the target
            y_trans = encode_target(y)
            # concatenate the data
            X_trans[TARGET] = y_trans
            logger.log_message(f'Shape of the final dataframe is {X_trans.shape}')
            # save the data
            save_the_data(X_trans,save_data_path / 'test_final.csv')

if __name__ == "__main__":
    main()