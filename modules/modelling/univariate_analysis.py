import pandas as pd
from statsmodels.regression.linear_model import OLS
from .data_preparation import load_data_modelling
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from scipy.stats import ttest_rel

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
    
def univariate_analysis(x_data: pd.DataFrame, y_data:pd.DataFrame):
    results = {}
    x_norm = (x_data-x_data.mean(axis=None))/x_data.std()
    for c in y_data.columns:
        y =np.reshape(y_data[c], (x_data.shape[0], 1))
        model = OLS(y, x_norm)
        res = model.fit()
        table = np.round(pd.DataFrame(res.summary().tables[1].data), 4)
        table.columns = table.iloc[0,:]
        table = table.drop(index=0)
        table.index = table.iloc[:,0]
        table = table.drop(table.columns[0], axis=1)
        table = table.astype(float)
        results[c] = table, res.rsquared_adj, np.round(float(res.summary().tables[0][3][3].data),4)
        
    return results

def plot_significant_weights(model_results, output_path_base, only_components=[], title=""):
    labelmap = {
        "Behavioral_0": "Emotion regulation",
        "Behavioral_1": "Frustration",
        "Behavioral_2": "Anger",
        "Behavioral_3": "Working Memory",
        "Behavioral_4": "Anxiety",
        "Behavioral_5": "Social openness",
        "Behavioral_6": "Emotional reaction to task",
        "Behavioral_7": "Dietary restraint",
        "Behavioral_8": "Worrying control",
        "Behavioral_9": "Processing speed",
        "Behavioral_10": "Social orientation",
        "Behavioral_11": "Reappraisal",
        "Behavioral_12": "Impulsiveness",
        "Behavioral_13": "Verbal intelligence",
        "Behavioral_14": "Emotionality"
    }
    all_dfs = []
    perm_importances = {}
    i = 0
    for k, res in model_results.items():
        sub_df = res[0][res[0].iloc[:,3]<0.05].copy()
        label = labelmap[k] if k in labelmap.keys() else k
        perm_importances[label] = res[0]["coef"]**2
        # if not sub_df.empty and i in only_components:
        if not sub_df.empty:
            sub_df["Behavioral Component"] = label
            all_dfs.append(sub_df)
        i +=1
    df = pd.concat(all_dfs)

    # Plot the squared coefficients
    perm_importances = pd.DataFrame(perm_importances)
    sum_squares = perm_importances.sum(axis=1)
    fig, axes = plt.subplots(1,1)
    ax = sns.barplot(x=sum_squares, y=perm_importances.index, errorbar=None)
    ax.set(xlabel="Sum of squared weights")
    ax.set_title(title)
    plt.tight_layout(pad=1.)
    fig.savefig(output_path_base+"_importance.png", dpi=300)
    
    comps = df["Behavioral Component"].unique()
    n_cols = 2
    fig, axes = plt.subplots(len(comps)//n_cols + min(len(comps)%n_cols,1), n_cols, figsize=(n_cols*6, len(comps)*4/n_cols))
    axes = axes.flatten()
    for i,x in enumerate(comps):
        sub_df = df[df["Behavioral Component"] == x]
        t = textwrap.fill(x, width=45, break_long_words=False)
        sub_df_up = sub_df.copy()
        sub_df_down = sub_df.copy()
        sub_df_up["coef"] = sub_df["coef"] + sub_df["std err"]
        sub_df_down["coef"] = sub_df["coef"] - sub_df["std err"]
        sub_df = pd.concat([sub_df_up, sub_df_down])
        sns.pointplot(sub_df,x=sub_df["coef"], y=sub_df.index, ax=axes[i], color=sns.color_palette("husl")[0], linestyle="none", marker="o", capsize=.2).set_title(t,fontweight='bold')
        # wrap_labels(axes[i], 15)
    if len(comps)%n_cols != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout(pad=4.)
    fig.savefig(output_path_base+"_weights.png", dpi=300)

def compare_rsquares(res1, res2):
    x1 = [x[1] for x in res1.values()]
    x2 = [x[1] for x in res2.values()]
    print("Comparing distribution of r squares:")
    print(ttest_rel(x1, x2))

def pipeline_univariate_analysis(all_data_path:str, output_path:str):
    behavioral_data, data_canica, data_dictionary_learn = load_data_modelling(all_data_path)
    canica_results = univariate_analysis(data_canica, behavioral_data)
    dictionary_learn_results = univariate_analysis(data_dictionary_learn, behavioral_data)
    compare_rsquares(canica_results, dictionary_learn_results)

    plot_significant_weights(canica_results, os.path.join(output_path, "univariate_canica"),[3,6], "CanICA")
    with pd.ExcelWriter(os.path.join(output_path, "univariate_results_canica.xlsx"), engine="openpyxl") as writer:
        r_square_results = []
        for k, res in canica_results.items():
            res[0].to_excel(writer, sheet_name=k, index=None)
            r_square_results.append((k,res[1], res[2]))
        r_square_df = pd.DataFrame(r_square_results, columns=["Behavioral Component", "R2 Canica", "p-value Canica"])
        r_square_df.to_excel(writer, sheet_name="R_Square", index = None)

    plot_significant_weights(dictionary_learn_results, os.path.join(output_path, "univariate_dictionary_learn"), [4, 5], "Dictionary Learn")
    with pd.ExcelWriter(os.path.join(output_path, "univariate_results_dictionary_learn.xlsx"), engine="openpyxl") as writer:
        r_square_results = []
        for k, res in dictionary_learn_results.items():
            res[0].to_excel(writer, sheet_name=k, index=None)
            r_square_results.append((k,res[1], res[2]))
        r_square_df = pd.DataFrame(r_square_results, columns=["Behavioral Component", "R2 DL", "p-value DL"])
        r_square_df.to_excel(writer, sheet_name="R_Square", index = None)