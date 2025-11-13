import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


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


def remove_outliers_iqr(data, cols):
    clean_data = data.copy()
    for col in cols:
        if clean_data[col].nunique() > 10:  # skip discrete columns
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Replace the outliers with upper and lower limit
            clean_data[col] = np.where(clean_data[col] < lower, lower, clean_data[col])
            clean_data[col] = np.where(clean_data[col] > upper, upper, clean_data[col])

    return clean_data


def restore_original_dtypes(df, dtype_map):
    for col, original_dtype in dtype_map.items():
        if col in df.columns:
            try:
                if pd.api.types.is_integer_dtype(original_dtype):
                    df[col] = df[col].round().astype('int64')
                elif pd.api.types.is_categorical_dtype(original_dtype):
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(original_dtype)
            except Exception as e:
                print(f"Could not restore dtype for '{col}' : {e}")
    return df



def get_clean_dataframe(dataframes, df_names):

    # Dictionary of column name for each dataframe for rename
    column_names = {
        0:{
            '0': "customer_id",
            '1': "gender",
            '2': "age",
            '3': "driving_licence_present",
            '4': "region_code",
            '5': "previously_insured",
            '6': "vehicle_age",
            '7': "vehicle_damage"
        },
        1:{
            '0': "customer_id",
            '1': "annual_premium [in Rs]",
            '2': "sales_channel_code",
            '3': "vintage",
            '4': "response"
        }
    }

    # rename the columns
    for idx, name in enumerate(df_names):
        dataframes[name].rename(columns=column_names[idx], inplace=True)

    # Assign the first df by indexing
    df = dataframes[df_names[0]]
    
    # Merge the multiple dataframes into one single datarame
    for i in range(1, len(df_names)):
        df = pd.merge(df, dataframes[df_names[i]], on="customer_id", how="inner")
    
    # Remove records where customer_id is null
    df = df[df['customer_id'].notnull()]

    # create list of numeric columns
    numeric_cols = [
        "customer_id", "age", "driving_licence_present",
        "region_code", "previously_insured",
        "annual_premium [in Rs]", "sales_channel_code",
        "vintage", "response"
    ]

    # create list of categorical columns
    categorical_cols = ["gender", "vehicle_age", "vehicle_damage"]

    # Replace nulls in numeric columns with mean
    for col in numeric_cols:
        if col in df.columns:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)

    # Replace nulls in categorical columns with mode
    for col in categorical_cols:
        if col in df.columns:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_value, inplace=True)
    
    # Dictionary of columns with their corresponding datatype
    dtype_map = {
        "customer_id": "int64",
        "gender": "category",
        "age": "int64",
        "driving_licence_present": "int64",
        "region_code": "int64",
        "previously_insured": "int64",
        "vehicle_age": "category",
        "vehicle_damage": "category",
        "annual_premium [in Rs]": "float64",
        "sales_channel_code": "int64",
        "vintage": "int64",
        "response": "int64"
    }

    # Convert the columns datatype
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)  
            except Exception:
                pass

    # Encode 'vehicle_age' values                
    vehicle_age_map = {
        "< 1 Year": 1,
        "1-2 Year": 2,
        "> 2 Years": 3
    }
    
    df["vehicle_age"] = df["vehicle_age"].replace(vehicle_age_map)
    # Convert encoded values to numeric type
    df["vehicle_age"] = pd.to_numeric(df["vehicle_age"], errors='coerce').fillna(0).astype(int)

    original_dtypes = {col: df[col].dtype for col in df.columns}

    df = remove_outliers_iqr(df, ["age", "annual_premium [in Rs]", "vintage"])
    clean_df = restore_original_dtypes(df, original_dtypes)
    # Reset index
    clean_df.reset_index(drop=True, inplace=True)

    return clean_df


def main():
    dfs, names = get_dataframes()
    final_df = get_clean_dataframe(dfs, names)
    print(final_df.info())
    print(final_df.head(5))
    print("Rows:",final_df.shape[0])
    print("Columns:",final_df.shape[1])

if __name__ == "__main__":
    main()
