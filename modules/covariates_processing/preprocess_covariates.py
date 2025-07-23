from glob import glob
import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import re
from ..mri_processing.load_mri import read_dual_regression_data

# Load functions
def load_data_file(path:str) -> pd.DataFrame:
    if path.lower().endswith("csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith("xlsx"):
        df = pd.read_excel(path)
    else:
        raise f"{path} doesn't have a supported extension"
    return df

def covariates_data_loader(path:str) -> dict:
    omit_tests = ["MDBF_Day1.csv", "MDBF_Day2.csv", "MDBF_Day3.csv", "FTP.csv", "YFAS.csv", "MARS.csv"]
    get_columns = {
        "CVLT.csv": ["CVLT_3", "CVLT_4", "CVLT_5", "CVLT_6" ,"CVLT_8", "CVLT_11" ,"CVLT_13"],
        "TAP-Alertness.csv": ["TAP_A_7", "TAP_A_9", "TAP_A_12", "TAP_A_14"],
        "TAP-Incompatibility.csv": ["TAP_I_10", "TAP_I_12", "TAP_I_14", "TAP_I_17", "TAP_I_19", "TAP_I_21"],
        "TAP-Working Memory.csv": ["TAP_WM_3", "TAP_WM_5", "TAP_WM_8", "TAP_WM_10"],
        "RWT.csv": ["RWT_9", "RWT_11", "RWT_21", "RWT_23"],
        "TMT.csv": ["TMT_1", "TMT_3", "TMT_5", "TMT_7"],
        "WST.csv": ["WST_3"],
        "MSPSS.csv": ["MSPSS_total"],
        "PSQ.csv": ["PSQ_OverallScore"],
        "TAS.csv": ["TAS_OverallScore"],
        "LOT-R.csv": ["LOT_Optimism", "LOT_Pessimism"]
    }
    if os.path.isfile(path):
        fname = os.path.basename(path)
        return {fname: read_single_covariates_file(path)}
    else:
        covariates = {}
        for f in glob(f"{path}/**/*.csv", recursive=True):
            fname = os.path.basename(f)
            if fname in omit_tests:
                continue
            new_df = read_single_covariates_file(f)
            if fname in get_columns.keys():
                new_df = new_df[get_columns[fname]]
            covariates[fname] = new_df
        return covariates

def read_single_covariates_file(path:str):
    replace_values = {
        "<1": 0,
        ">76": 80,
        ">99": 100,
        '>98': 100,
        '>50': 50,
        " ": float("nan"),
        "*": float("nan")
    }
    df = load_data_file(path)
    df = df.rename(columns={"Unnamed: 0": "subject"})   
    df = df.set_index("subject")
    df = df.replace(replace_values)
    return df

# Cleaning noise data functions
def clean_missing_data(df: pd.DataFrame):
    df.columns = df.columns.droplevel(0)
    # Filter columns by missing data
    thr_cols = 0.05*df.shape[0]
    column_nans = df.isna().sum(axis=0)
    noise_columns = column_nans[column_nans > thr_cols]
    # print(column_nans)
    print(f"The following columns will be removed:\n{noise_columns}")
    df = df.drop(columns=noise_columns.index)

    # Detect subjects with a lot of missing data
    thr_subjects = 0.10*df.shape[1]
    subject_nans = df.isna().sum(axis=1)
    noise_subjects = subject_nans[subject_nans > thr_subjects]
    # print(subject_nans)
    print(f"The following subjects will be removed: {noise_subjects}")
    df = df.drop(index=noise_subjects.index)

    # Impute the missing data
    imp_mean = IterativeImputer(random_state=0)
    df.loc[:,:] = imp_mean.fit_transform(df)
    # Filter columns with very low variability
    column_std = df.std(axis=0)
    print(f"Colums with a low std: {column_std[column_std < 0.2]}")
    return df

# Preprocessing functions
def sum_columns(df:pd.DataFrame, col_prefix):
    columns = [x for x in df.columns if col_prefix in x]
    res_series = df[columns].sum(axis=1)
    df = df.drop(columns=columns, inplace=True)
    return res_series

# From https://doi.org/10.1177/01939459211012044
def transform_COPE(df:pd.DataFrame):
    # Disengage
    disengage_columns = ["COPE_BehavioralDisengagement", "COPE_Denial", "COPE_SelfBlame", "COPE_Venting", "COPE_SelfDistraction"]
    df["COPE_disengage"] = df[disengage_columns].sum(axis=1)
    df = df.drop(columns=disengage_columns)
    # Social support
    social_columns = ["COPE_UseOfEmotionalSupport", "COPE_UseOfInstrumentalSupport"]
    df["COPE_social_support"] = df[social_columns].sum(axis=1)
    df = df.drop(columns=social_columns)
    # Active
    active_columns = ["COPE_activeCoping", "COPE_positiveReframing", "COPE_Humor", "COPE_Planning", "COPE_Alkohol\u200e_Drogen", "COPE_Acceptance", "COPE_Religion"]
    df["COPE_active"] = df[active_columns].sum(axis=1)
    df = df.drop(columns=active_columns)
    return df

def transform_TICS(df:pd.DataFrame):
    # Work stress
    work_columns = ["TICS_WorkOverload", "TICS_WorkDiscontent", "TICS_WorkDemands"]
    df["TICS_Work_Stress"] = df[work_columns].sum(axis=1)
    df = df.drop(columns=work_columns)
    # Social stress
    social_columns = ["TICS_SocialOverload", "TICS_SocialTension", "TICS_SocialIsolation"]
    df["TICS_Social_Stress"] = df[social_columns].sum(axis=1)
    df = df.drop(columns=social_columns+["TICS_ScreeningScale"])
    return df


def transform_columns(df:pd.DataFrame):
    # Sum of NYC subscales
    df["NYC-Q_lemon_sum"] = sum_columns(df, "NYC-Q_lemon")
    # Sum of BAS subscales
    df["BAS_sum"] = sum_columns(df, "BAS_")
    # Factorize COPE variables
    df = transform_COPE(df)
    df = transform_TICS(df)
    return df

def load_and_preprocess_covariates(cognitive_test_folder:str, personality_test_folder:str) -> pd.DataFrame:
    """Main function to load the covariates

    Args:
        cognitive_test_folder (str): folder containing the results from cognitive batteries
        personality_test_folder (str): folder containing the results from personality batteries

    Returns:
        pd.DataFrame: Dataframe with the preprocessed covariates
    """
    cognitive_tests = covariates_data_loader(cognitive_test_folder)
    personality_tests = covariates_data_loader(personality_test_folder)

    cognitive_tests_joined = pd.concat(cognitive_tests, axis=1)
    personality_tests_joined = pd.concat(personality_tests, axis=1)
    all_data = pd.concat([cognitive_tests_joined, personality_tests_joined], axis=1)
    clean_all_data = clean_missing_data(all_data)
    clean_all_data = transform_columns(clean_all_data)
    print(f"{(len(cognitive_tests.keys()))} cognitive tests loaded:\n{list(cognitive_tests.keys())}")
    print(f"{(len(personality_tests.keys()))} cognitive tests loaded:\n{list(personality_tests.keys())}")
    return clean_all_data

def load_and_preprocess_metadata(metadata_path:str) -> pd.DataFrame:
    metadata = read_single_covariates_file(metadata_path)
    metadata["Gender"] = metadata["Gender_ 1=female_2=male"].replace({2: 1, 1: 0})
    metadata["Age"] = metadata["Age"].replace({
        "20-25": 0,
        "25-30": 1,
        "30-35": 2,
        "35-40": 3,
        "40-45": 4,
        "45-50": 5,
        "50-55": 6,
        "55-60": 7,
        "60-65": 8,
        "65-70": 9,
        "70-75": 10,
        "75-80": 11,
        "80-85": 12,
        "85-90": 13,
        "90-95": 14,
        "95-100": 15
    })
    metadata["Smoking"] = metadata["Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)"].replace({
        "0": 0, "1": 1, "2": 2
    })
    metadata["Alcohol"] = metadata["Standard_Alcoholunits_Last_28days"].apply(lambda x: eval(x.replace(",", ".")) if x==x else float("nan"))
    metadata = metadata[["Gender", "Age", "Smoking", "Alcohol"]]
    return metadata

def join_mri_covariates(func_paths_train: list, func_paths_test: list, canica_output_path:str, 
                        dictionary_learn_output_path:str, covariates: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    subjects_train = [x for x in covariates.index if any([re.search(x, y.get_filename()) for y in func_paths_train])]
    subjects_test = [x for x in covariates.index if any([re.search(x, y.get_filename()) for y in func_paths_test])]
    mri_df = pd.DataFrame({"split": ["test"]*len(subjects_test) + ["train"]*len(subjects_train)}, index = subjects_test+subjects_train)
    mri_df = read_dual_regression_data(mri_df, os.path.join(canica_output_path, "dual_regression"), "canica")
    mri_df = read_dual_regression_data(mri_df, os.path.join(dictionary_learn_output_path, "dual_regression"), "dictionary_learn")
    all_data = pd.concat([covariates, metadata, mri_df], axis=1, join="inner")
    return all_data