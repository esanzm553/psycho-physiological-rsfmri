import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.stats.multitest import multipletests
from .data_preparation import load_data_modelling
import os

def multivariate_analysis(x_data: pd.DataFrame, y_data:pd.DataFrame):
    res = MANOVA(y_data, x_data).mv_test()
    r = CanCorr(y_data, x_data)
    res.endog_names = y_data.columns
    res.exog_names = x_data.columns
    tables = [x.iloc[0,1:] for i,x in enumerate(res.summary().tables) if i%2!=0]
    tables = pd.DataFrame(tables, index=x_data.columns)
    tables["Pr > F"] = multipletests(tables["Pr > F"])[1]
    mv_tables = tables.round(4)
    # tables = {x_data.columns[i]:x for i,x in enumerate(tables)}
    cca_table = pd.DataFrame(r.corr_test().summary().tables[-1])
    return mv_tables, cca_table

def pipeline_multivariate_analysis(all_data_path:str, output_path:str):
    
    x_data, y_data_canica, y_data_dictionary_learn = load_data_modelling(all_data_path)
    canica_results_mv_psy_phy, canica_results_cca_psy_phy = multivariate_analysis(x_data, y_data_canica)
    canica_results_mv_phy_psy, canica_results_cca_phy_psy = multivariate_analysis(y_data_canica, x_data)
    dictionary_learn_results_mv_psy_phy, dictionary_learn_results_cca_psy_phy = multivariate_analysis(x_data, y_data_dictionary_learn)
    dictionary_learn_results_mv_phy_psy, dictionary_learn_results_cca_phy_psy = multivariate_analysis(y_data_dictionary_learn, x_data)
    
    with pd.ExcelWriter(os.path.join(output_path, "multivariate_results_canica.xlsx"), engine="openpyxl") as writer:
        canica_results_mv_psy_phy.to_excel(writer, sheet_name="MV_PSY_PHY")
        canica_results_mv_phy_psy.to_excel(writer, sheet_name="MV_PHY_PSY")
        canica_results_cca_psy_phy.to_excel(writer, sheet_name="CCA_PSY_PHY")
        canica_results_cca_phy_psy.to_excel(writer, sheet_name="CCA_PHY_PSY")

    with pd.ExcelWriter(os.path.join(output_path, "multivariate_results_dictionary_learn.xlsx"), engine="openpyxl") as writer:
        dictionary_learn_results_mv_phy_psy.to_excel(writer, sheet_name="MV_PHY_PSY")
        dictionary_learn_results_mv_psy_phy.to_excel(writer, sheet_name="MV_PSY_PHY")
        dictionary_learn_results_cca_psy_phy.to_excel(writer, sheet_name="CCA_PSY_PHY")
        dictionary_learn_results_cca_phy_psy.to_excel(writer, sheet_name="CCA_PHY_PSY")
