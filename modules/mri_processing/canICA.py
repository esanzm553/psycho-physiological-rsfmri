from nilearn.decomposition import CanICA
from .analyze_mri import dual_regression
import os
import logging
import pickle
from tqdm import tqdm
import numpy as np


def run_canICA(output_path:str, func_data_train:list, func_data_test:list|None=None, confounds=None, n_components=20, n_jobs:int=2, cache:str|None=None, mask_file:str|None=None):
    # Paths
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    canica_image_path = os.path.join(output_path, "canica_resting_state.nii.gz")
    canica_results_path = os.path.join(output_path, "canica.pickle")
    canica_scores_train_path = os.path.join(output_path, "canica_scores_train.pickle")
    canica_scores_test_path = os.path.join(output_path, "canica_scores_test.pickle")
    canica_projections_path = os.path.join(output_path, "dual_regression")
    os.makedirs(canica_projections_path, exist_ok=True)
    
    # CanICA fitting
    if os.path.exists(canica_results_path):
        logging.info(f"Loading previous CanICA model from {canica_results_path}...")
        with open(canica_results_path, 'rb') as handle:
            canica = pickle.load(handle)
    else:
        canica = CanICA(
            mask=mask_file,
            n_components=n_components,
            memory=cache,
            memory_level=1,
            verbose=5,
            random_state=0,
            standardize="zscore_sample",
            n_jobs=n_jobs,
        )
        logging.info("Fitting CanICA...")
        try:
            canica.fit(func_data_train, confounds=confounds)
        except Exception as e:
            logging.error(f"Error fitting the CanICA model: {str(e)}")
            raise e
        logging.info(f"Done. Writing the results in {canica_results_path}...")
        with open(canica_results_path, 'wb') as handle:
            pickle.dump(canica, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Done. Writing the image of the components in {canica_image_path}...")
        canica_components_img = canica.components_img_
        canica_components_img.to_filename(canica_image_path)

    # Scores train
    if os.path.exists(canica_scores_train_path):
        logging.info("Omitting already calculated train scores")
    else:
        scores_train = []
        scores_components_train = []
        logging.info(f"Calculating the explained variance in the train dataset....")
        for func_data_train_subject in tqdm(func_data_train):
            scores_train.append(canica.score(func_data_train_subject, confounds=None, per_component=False))
            scores_components_train.append(canica.score(func_data_train_subject, confounds=None, per_component=True))
        scores_train_mean = np.mean(scores_train)
        scores_train_std = np.std(scores_train)
        scores_components_train = np.array(scores_components_train)
        scores_components_train_mean = np.mean(scores_components_train, axis=0)
        scores_components_train_std = np.std(scores_components_train, axis=0)

        scores_train = {
            "scores_train": scores_train,
            "scores_train_mean": scores_train_mean,
            "scores_train_std": scores_train_std,
            "scores_components_train": scores_components_train,
            "scores_train_components_mean": scores_components_train_mean,
            "scores_train_components_std": scores_components_train_std,
        }
        with open(canica_scores_train_path, 'wb') as handle:
            pickle.dump(scores_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Scores test
    if os.path.exists(canica_scores_test_path):
        logging.info("Omitting already calculated test scores")
    else:
        scores_test = []
        scores_components_test = []
        logging.info(f"Calculating the explained variance in the test dataset....")
        for func_data_test_subject in tqdm(func_data_test):
            scores_test.append(canica.score(func_data_test_subject, confounds=None, per_component=False))
            scores_components_test.append(canica.score(func_data_test_subject, confounds=None, per_component=True))
        scores_test_mean = np.mean(scores_test)
        scores_test_std = np.std(scores_test)
        scores_components_test = np.array(scores_components_test)
        scores_components_test_mean = np.mean(scores_components_test, axis=0)
        scores_components_test_std = np.std(scores_components_test, axis=0)

        scores_test = {
            "scores_test": scores_test,
            "scores_test_mean": scores_test_mean,
            "scores_test_std": scores_test_std,
            "scores_components_test": scores_components_test,
            "scores_test_components_mean": scores_components_test_mean,
            "scores_test_components_std": scores_components_test_std,
        }
        with open(canica_scores_test_path, 'wb') as handle:
            pickle.dump(scores_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dual regression analysis
    for sub in func_data_train+func_data_test:
        fname = os.path.basename(sub.get_filename())
        sub_path_img = os.path.join(canica_projections_path, "dr_"+fname)
        sub_path_corr = os.path.join(canica_projections_path, "dr_"+fname.replace(".nii.gz", "_correlations.pickle"))
        if os.path.exists(sub_path_img) and os.path.exists(sub_path_corr):
            logging.info("Omitting already calculated dual_regression")
        else:
            logging.info(f"Calculating the dual regression of subject {fname}...")
            comp_img, correlations = dual_regression(sub, canica, mask_file)
            comp_img.to_filename(sub_path_img)
            with open(sub_path_corr, 'wb') as handle:
                pickle.dump(correlations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return canica