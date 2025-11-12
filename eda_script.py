import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


def get_dataframes():
    try:
        # check if folder exists
        data_path = Path(os.getcwd())/"data"
        if not data_path.exists():
            raise FileNotFoundError(f"'data' folder not found as: {data_path}")
        
        # get all csv files from data folder
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in the 'data' folder.")
        
        # load each csv into dataframe
        df_dict = {}
        for file in csv_files:
            df_name = file.stem
            try:
                df_dict[df_name] = pd.read_csv(file)
            except Exception as e:
                print(f"[Warning] Could not load {file.name} : {e}")
        
        # Create list of dataframe names
        df_names = list(df_dict.keys())

        return df_dict, df_names
    
    except Exception as e:
        print(f"[Error] {e}")

