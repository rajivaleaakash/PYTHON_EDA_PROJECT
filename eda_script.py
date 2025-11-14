import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from logging_setup import setup_logger

# Initialize logger for this script
logger = setup_logger("eda_script")


def get_dataframes():
    """
    Load all CSV files from the 'data' directory and return them as a dictionary.

    Returns:
        df_dict (dict): Dictionary containing dataframes loaded from CSV files.
        df_names (list): List of dataframe names (file stems).

    This function:
        - Validates existence of the 'data' directory.
        - Loads all CSVs into pandas DataFrames.
        - Logs loading progress.
        - Handles missing files or load failures gracefully.
    """
    logger.info("Starting to load CSV files from data directory...")
    try:
        data_path = Path(os.getcwd()) / "data"
        if not data_path.exists():
            raise FileNotFoundError(f"'data' folder not found at: {data_path}")

        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in the 'data' folder.")

        df_dict = {}
        for file in csv_files:
            try:
                df_dict[file.stem] = pd.read_csv(file)
                logger.info(f"Loaded file: {file.name} | Shape: {df_dict[file.stem].shape}")
            except Exception as e:
                logger.warning(f"Could not load {file.name} : {e}")

        df_names = list(df_dict.keys())
        logger.info(f"Successfully loaded {len(df_names)} dataframes.")
        return df_dict, df_names

    except Exception as e:
        logger.error(f"Error in get_dataframes(): {e}")
        return {}, []


def remove_outliers_iqr(data, cols):
    """
    Apply IQR-based outlier capping on numeric columns.

    Args:
        data (DataFrame): Input dataframe.
        cols (list): List of numeric columns to process.

    Returns:
        DataFrame: Outlier-capped dataframe.

    This function:
        - Calculates Q1, Q3, IQR for each numeric column.
        - Computes lower & upper outlier thresholds.
        - Replaces outliers with threshold values (winsorization).
    """
    logger.info("Running IQR outlier capping...")
    clean_data = data.copy()

    for col in cols:
        try:
            # Avoid capping for discrete integer-coded columns
            if clean_data[col].nunique() > 10:
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                logger.info(
                    f"[{col}] Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
                    f"Lower={lower:.2f}, Upper={upper:.2f}"
                )

                # Cap the outliers
                clean_data[col] = np.where(clean_data[col] < lower, lower, clean_data[col])
                clean_data[col] = np.where(clean_data[col] > upper, upper, clean_data[col])

        except Exception as e:
            logger.error(f"Failed processing outliers for '{col}': {e}")

    return clean_data


def restore_original_dtypes(df, dtype_map):
    """
    Restore original datatypes after transformations.

    Args:
        df (DataFrame): Modified dataframe.
        dtype_map (dict): Original datatypes before modification.

    Returns:
        DataFrame: Dataframe with restored original dtypes.

    This ensures:
        - Integer columns stay int (not accidentally converted to float).
        - Categorical columns are restored as categories.
        - Other columns return to original dtype.
    """
    logger.info("Restoring original column data types...")
    for col, original_dtype in dtype_map.items():
        try:
            if pd.api.types.is_integer_dtype(original_dtype):
                df[col] = df[col].round().astype("int64")
            elif pd.api.types.is_categorical_dtype(original_dtype):
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(original_dtype)

        except Exception as e:
            logger.warning(f"Could not restore dtype for '{col}': {e}")

    logger.info("Datatype restoration complete.")
    return df


def save_clean_df(df, name="clean_merge_data"):
    """
    Save cleaned dataframe in two formats:
        1. A timestamped version (version history)
        2. A master cumulative dataset (clean_master_data.csv)

    Args:
        df (DataFrame): Clean dataframe to save.
        name (str): Base name for versioned files.

    This function:
        - Ensures directory exists.
        - Creates version-controlled snapshots.
        - Maintains a growing master dataset.
    """
    logger.info("Saving cleaned dataframe...")

    try:
        save_dir = Path("clean_merge_dataset")
        save_dir.mkdir(exist_ok=True)

        # File naming
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamped_file = save_dir / f"{name}_{timestamp}.csv"
        master_file = save_dir / "clean_master_data.csv"

        # Save timestamped version
        df.to_csv(timestamped_file, index=False)
        logger.info(f"Timestamped file saved: {timestamped_file}")

        # Save or update master file
        if master_file.exists():
            logger.info(f"Appending to master file: {master_file}")
            existing_df = pd.read_csv(master_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
            combined_df.to_csv(master_file, index=False)
        else:
            logger.info(f"Creating new master file: {master_file}")
            df.to_csv(master_file, index=False)

        logger.info("Clean dataset successfully saved.")

    except Exception as e:
        logger.error(f"Error in save_clean_df(): {e}")


def get_clean_dataframe(dataframes, df_names):
    """
    Clean, merge, encode, and process multiple raw insurance datasets.

    Args:
        dataframes (dict): Dictionary of loaded dataframes.
        df_names (list): Names of dataframes in merge order.

    Returns:
        DataFrame: Fully cleaned and merged dataset.

    Processing steps include:
        1. Column renaming
        2. Inner merging on customer_id
        3. Missing-value handling
        4. Datatype enforcement
        5. Category encoding (vehicle_age)
        6. IQR-based outlier capping
        7. Restoration of original data types
    """
    logger.info("Starting data cleaning & merging pipeline...")

    try:
        # Column name mapping per dataframe
        column_names = {
            0: {
                '0': "customer_id",
                '1': "gender",
                '2': "age",
                '3': "driving_licence_present",
                '4': "region_code",
                '5': "previously_insured",
                '6': "vehicle_age",
                '7': "vehicle_damage"
            },
            1: {
                '0': "customer_id",
                '1': "annual_premium [in Rs]",
                '2': "sales_channel_code",
                '3': "vintage",
                '4': "response"
            }
        }

        # Rename columns according to mapping
        for idx, name in enumerate(df_names):
            dataframes[name].rename(columns=column_names[idx], inplace=True)
            logger.info(f"Columns renamed for dataframe: {name}")

        # Merge dataframes sequentially on customer_id
        df = dataframes[df_names[0]]
        for i in range(1, len(df_names)):
            df = pd.merge(df, dataframes[df_names[i]], on="customer_id", how="inner")

        logger.info(f"Data merged successfully | Shape: {df.shape}")

        df = df[df["customer_id"].notnull()]

        # Column groups
        numeric_cols = [
            "customer_id", "age", "driving_licence_present",
            "region_code", "previously_insured",
            "annual_premium [in Rs]", "sales_channel_code",
            "vintage", "response"
        ]

        categorical_cols = ["gender", "vehicle_age", "vehicle_damage"]

        # Fill numeric missing values
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        # Fill categorical missing values
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Expected datatype mapping
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

        # Apply datatype rules
        for col, dtype in dtype_map.items():
            try:
                df[col] = df[col].astype(dtype)
            except:
                pass

        # Encode vehicle age
        vehicle_age_map = {
            "< 1 Year": 1,
            "1-2 Year": 2,
            "> 2 Years": 3
        }
        df["vehicle_age"] = df["vehicle_age"].replace(vehicle_age_map)
        df["vehicle_age"] = pd.to_numeric(df["vehicle_age"], errors="coerce").fillna(0).astype(int)

        # Backup original column dtypes
        original_dtypes = {col: df[col].dtype for col in df.columns}

        # Outlier capping
        df = remove_outliers_iqr(df, ["age", "annual_premium [in Rs]", "vintage"])

        # Restore original integer & categorical dtypes
        df = restore_original_dtypes(df, original_dtypes)

        df.reset_index(drop=True, inplace=True)
        logger.info("Data cleaning completed successfully.")

        return df

    except Exception as e:
        logger.error(f"Error in get_clean_dataframe(): {e}")


def main():
    """
    Main execution workflow for the EDA pipeline.

    Workflow:
        1. Load raw CSV datasets.
        2. Clean and merge them.
        3. Log dataframe stats.
        4. Save output in both timestamped and master formats.
    """
    logger.info("----- EDA Script Execution Started -----")

    dfs, names = get_dataframes()

    if not dfs:
        logger.error("No dataframes loaded. Exiting process.")
        return

    final_df = get_clean_dataframe(dfs, names)

    logger.info(f"Final dataframe shape: {final_df.shape}")
    save_clean_df(final_df)

    logger.info("----- EDA Script Execution Completed -----")


if __name__ == "__main__":
    main()
