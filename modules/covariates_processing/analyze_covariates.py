import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import Isomap, SpectralEmbedding, MDS, trustworthiness
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_style("darkgrid")

def optim_reduce_components(df:pd.DataFrame, model, title, graphics_path) -> pd.DataFrame:
    normalized_df = minmax_scale(df, axis=0)
    X, Y = train_test_split(normalized_df, test_size=0.2, random_state=777)
    worths_x = []
    worths_y = []
    max_comp = 5
    for n_comp in range(5,40):
        try:
            embedding = model(n_components=n_comp)
            X_transformed = embedding.fit_transform(X)
            Y_transformed = embedding.fit_transform(Y)
            worth_X = trustworthiness(X, X_transformed)
            worth_Y = trustworthiness(Y, Y_transformed)
            worths_x.append(worth_X)
            worths_y.append(worth_Y)
        except:
            print(title, "stopping at ", n_comp)
            break
        max_comp += 1
    
    worth_df = pd.DataFrame({"N Components": list(range(5,max_comp))*2, "Trustworthiness": worths_x+worths_y, "dataset": ["train"]*(max_comp-5)+["test"]*(max_comp-5)})
    # Plot the ROC graphics
    plt.figure()
    ax = sns.lineplot(data=worth_df, x="N Components", y="Trustworthiness", hue="dataset")
    ax.set_title(title)
    ax.set_xlabel(ax.get_xlabel(), fontdict={'weight': 'bold'})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'weight': 'bold'})
    plt.savefig(os.path.join(graphics_path, title+".png"))
    return worth_df

def reduce_components(df:pd.DataFrame) -> tuple[MDS, np.ndarray]:
    normalized_df = minmax_scale(df, axis=0)
    embedding = MDS(n_components=15)
    return embedding, embedding.fit_transform(normalized_df)
    
def optimize_and_reduce_covariates(clean_all_data:pd.DataFrame, graphics_path:str, output_path:str) -> tuple[MDS, np.ndarray]:
    worth_isomap = optim_reduce_components(clean_all_data, Isomap, "Isomap", graphics_path)
    worth_spectral = optim_reduce_components(clean_all_data, SpectralEmbedding, "Spectal Embedding", graphics_path)
    worth_MDS = optim_reduce_components(clean_all_data, MDS, "MDS", graphics_path)
    embedding_model, components = reduce_components(clean_all_data)
    with pd.ExcelWriter(os.path.join(output_path, "trustworthiness_comparison.xlsx"), engine="openpyxl") as writer:
        worth_isomap.to_excel(writer, index=None, sheet_name="Isomap")
        worth_spectral.to_excel(writer, index=None, sheet_name="Spectral")
        worth_MDS.to_excel(writer, index=None, sheet_name="MDS")
    components = pd.DataFrame(data=components, index=clean_all_data.index, columns=["Behavioral_"+str(i) for i in range(components.shape[1])])
    return embedding_model, components

def calculate_feature_importances(clean_all_data:pd.DataFrame, components_df:pd.DataFrame, graphics_path:str, output_path:str) -> None:
    components = components_df.values
    with pd.ExcelWriter(os.path.join(output_path, "reduced_components.xlsx"), engine="openpyxl") as writer:
        df_print_comps = []
        for i in range(components.shape[1]):
            regressor = GradientBoostingRegressor()
            regressor.fit(clean_all_data, components[:,i])
            score = regressor.score(clean_all_data, components[:,i])
            # print(f"Component {i}: {score}; {sorted(enumerate(regressor.feature_importances_), key= lambda x: x[-1], reverse=True)[:5]}")
            r = permutation_importance(regressor, clean_all_data, components[:,i], n_repeats=15, n_jobs=8, scoring="r2")
            sorted_importance = sorted(enumerate(r['importances_mean']), key= lambda x: x[-1], reverse=True)
            sorted_importance_means = [x[1] for x in sorted_importance]
            sorted_importance_components = [x[0] for x in sorted_importance]
            sorted_importance_components_names = [clean_all_data.columns[x] for x in sorted_importance_components]
            sorted_importance_std = [r['importances_std'][x] for x in sorted_importance_components]
            cum_sorted_importance = np.cumsum(sorted_importance_means)

            # Plot results
            fig, ax = plt.subplots()
            max_elems = 10
            forest_importances = pd.Series(sorted_importance_means[:max_elems], index=sorted_importance_components_names[:max_elems])
            forest_importances.plot.bar(yerr=sorted_importance_std[max_elems], ax=ax)
            ax.set_title(f"Component {i}")
            ax.set_ylabel("Feature Importance")
            fig.tight_layout()
            plt.show()
            forest_importances_df = []
            forest_importances = forest_importances[forest_importances > 0.03]
            for k, x in enumerate(forest_importances.index):
                s = x.split("_")
                forest_importances_df.append({"Scale": " ".join(s[1:]), "Test": s[0], "Mean Importance": round(forest_importances.iloc[k],3)})
            forest_importances_df = pd.DataFrame(forest_importances_df)
            df_print_comps.append({"Component": f"Component {i}", "text": ", ".join([f"{x[1]['Scale']} ({x[1]['Test']})" for x in forest_importances_df.iterrows()])})
            forest_importances_df.to_excel(writer, index=None, sheet_name=str(i))
            plt.savefig(os.path.join(graphics_path, f"Component_{str(i)}.png"))
            plt.close()
        pd.DataFrame(df_print_comps).to_excel(writer, index=None, sheet_name="summary")