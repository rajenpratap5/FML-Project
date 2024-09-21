import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from yaml import safe_load
from MLOps.logger import CustomLogger
from typing import Tuple
import logging

# create the custom logger
logger = CustomLogger(logger_name='data_ingestion')

# create a stream handler
console_handler = logging.StreamHandler()

# add console handler to the logger
logger.logger.addHandler(console_handler)

# column to drop
COLUMN_TO_DROP = 'case_id'

# target column name
TARGET = 'case_status'

# url for data
URL = r'https://raw.githubusercontent.com/rajenpratap5/usvisa_jupyteranalysis/refs/heads/main/Visadataset.csv'




def load_data_from_link(url: str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(url)
        logger.log_message('dataset downloaded from url')
        
        return df
    except FileNotFoundError as e:
        logger.log_message('Unable to download dataset')



def do_data_splitting(dataframe: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    try:
        parameters = safe_load(open('params.yaml','r'))['data_ingestion']
        logger.log_message('Parameters read successfully')
    except FileNotFoundError as e:
        logger.log_message('Parameters file missing')
    
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logger.log_message(f'Parameters : test_size={test_size}  random_state={random_state}')
    
    train_data, test_data = train_test_split(dataframe,test_size=test_size,random_state=random_state)
    logger.log_message(f"""
                       Data split into train data with shape {train_data.shape} and 
                       test data with shape {test_data.shape}
                       """)   
     
    return train_data, test_data
    

def save_the_data(dataframe: pd.DataFrame, data_path: Path) -> None:
    try:
        # save the data to path
        dataframe.to_csv(data_path,index=False)
        logger.log_message(f"DataFrame {data_path.name} saved successfuly")
    except Exception as e:
        logger.log_message(f'Dataframe not saved')
        

            
def drop_column(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return (
        dataframe.drop(columns=[column_name])
    )
    
    

    
def main():
    # load the data from url
    df = load_data_from_link(URL)
    # drop the column from data
    df_trans = drop_column(df,COLUMN_TO_DROP)
    logger.log_message(f"Column name {COLUMN_TO_DROP} dropped from data")
    
    # split the data
    train_df, test_df = do_data_splitting(df_trans)
    
    # save the train and test data to a directory
    root_path = Path(__file__).parent.parent.parent
    # data directory
    data_path = root_path / "data" / "raw"
    # make the raw data directory
    data_path.mkdir(exist_ok=True,parents=True)
    logger.log_message("Raw data folder created")
    # save the train data
    save_the_data(train_df,data_path / "train.csv")
    # save the test data
    save_the_data(test_df,data_path / "test.csv")


    
if __name__ == "__main__":
    main()