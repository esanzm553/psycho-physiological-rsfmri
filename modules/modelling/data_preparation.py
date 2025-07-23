import pandas as pd
from ..mri_processing.load_mri import labelmaps
import statsmodels.api as sm


# Function to get the residuals of the datasets corrected by the covariates
def covariate_correction(df_variables, df_covariates):
    X = sm.add_constant(df_covariates)
    residuals = pd.DataFrame(index=df_variables.index)
    for column in df_variables.columns:
        y = df_variables[column]  
        modelo = sm.OLS(y, X).fit()
        y_pred = modelo.predict(X)
        residuals[column] = y - y_pred
    return residuals


def load_data_modelling(all_data_path, split=None):
    # Load data
    all_data_df = pd.read_excel(all_data_path, index_col=0)
    all_data_df = all_data_df.dropna(axis=0)
    # Select the split
    if split:
        all_data_df = all_data_df[all_data_df["split"] == split]
        all_data_df = all_data_df.drop(columns=["split"])
    else:
        all_data_df = all_data_df.drop(columns="split")
    # Divide all the data in a dataset for each ICA method, the covariates and the behavioral data
    canica_columns = [x for x in all_data_df.columns if "canica" in x]
    dictionary_learn_columns = [x for x in all_data_df.columns if "dictionary_learn" in x]
    canica_df = all_data_df[canica_columns]
    canica_df.columns = [labelmaps["canica"][int(x.split("_")[-1])] for x in canica_columns]
    dictionary_learn_df = all_data_df[dictionary_learn_columns]
    dictionary_learn_df.columns = [labelmaps["dictionary_learn"][int(x.split("_")[-1])] for x in dictionary_learn_columns]
    covariates = all_data_df[["Gender", "Age", "Smoking", "Alcohol"]]
    x_df = all_data_df.drop(columns=canica_columns+dictionary_learn_columns+covariates.columns.tolist())
    # Correct by covariates
    canica_df = covariate_correction(canica_df, covariates)
    dictionary_learn_df = covariate_correction(dictionary_learn_df, covariates)
    x_df = covariate_correction(x_df, covariates)
    return x_df, canica_df, dictionary_learn_df