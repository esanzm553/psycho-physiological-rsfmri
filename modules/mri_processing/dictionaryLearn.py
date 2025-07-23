from nilearn.decomposition import DictLearning
import os
import logging
from .analyze_mri import dual_regression
import pickle
from tqdm import tqdm
import numpy as np


def run_dictionary_learn(output_path:str, func_data_train:list, func_data_test:list|None=None, confounds=None, n_components=20, n_jobs:int=2, cache:str|None=None, mask_file:str|None=None):
    # Paths
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    dictionary_learn_image_path = os.path.join(output_path, "dictionary_learn_resting_state.nii.gz")
    dictionary_learn_results_path = os.path.join(output_path, "dictionary_learn.pickle")
    dictionary_learn_scores_train_path = os.path.join(output_path, "dictionary_learn_scores_train.pickle")
    dictionary_learn_scores_test_path = os.path.join(output_path, "dictionary_learn_scores_test.pickle")
    dictionary_learn_projections_path = os.path.join(output_path, "dual_regression")
    os.makedirs(dictionary_learn_projections_path, exist_ok=True)
    
    # dictionary_learn fitting
    if os.path.exists(dictionary_learn_results_path):
        logging.info(f"Loading previous dictionary_learn model from {dictionary_learn_results_path}...")
        with open(dictionary_learn_results_path, 'rb') as handle:
            dictionary_learn = pickle.load(handle)
    else:
        dictionary_learn = DictLearning(
            mask=mask_file,
            n_components=n_components,
            memory=cache,
            memory_level=1,
            verbose=5,
            random_state=0,
            standardize="zscore_sample",
            n_jobs=n_jobs,
        )
        logging.info("Fitting Dictionary Learn...")
        try:
            dictionary_learn.fit(func_data_train, confounds=confounds)
        except Exception as e:
            logging.error(f"Error fitting the Dictionary Learn model: {str(e)}")
            raise e
        logging.info(f"Done. Writing the results in {dictionary_learn_results_path}...")
        with open(dictionary_learn_results_path, 'wb') as handle:
            pickle.dump(dictionary_learn, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Done. Writing the image of the components in {dictionary_learn_image_path}...")
        dictionary_learn_components_img = dictionary_learn.components_img_
        dictionary_learn_components_img.to_filename(dictionary_learn_image_path)

    # Scores train
    if os.path.exists(dictionary_learn_scores_train_path):
        logging.info("Omitting already calculated train scores")
    else:
        scores_train = []
        scores_components_train = []
        logging.info(f"Done. Calculating the explained variance in the train dataset....")
        for func_data_train_subject in tqdm(func_data_train):
            scores_train.append(dictionary_learn.score(func_data_train_subject, confounds=None, per_component=False))
            scores_components_train.append(dictionary_learn.score(func_data_train_subject, confounds=None, per_component=True))
        scores_train_mean = np.mean(scores_train)
        scores_train_std = np.std(scores_train)
        scores_components_train = np.array(scores_components_train)
        scores_components_train_mean = np.mean(scores_components_train, axis=0)
        scores_components_train_std = np.std(scores_components_train, axis=0)

        scores_train = {
            "scores_train": scores_train,
            "scores_train_mean": scores_train_mean,
            "scores_train_std": scores_train_std,
            "scores_train_components": scores_components_train,
            "scores_train_components_mean": scores_components_train_mean,
            "scores_train_components_std": scores_components_train_std,
        }
        print(scores_train)
        with open(dictionary_learn_scores_train_path, 'wb') as handle:
            pickle.dump(scores_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Scores test
    if os.path.exists(dictionary_learn_scores_test_path):
        logging.info("Omitting already calculated test scores")
    else:
        scores_test = []
        scores_components_test = []
        logging.info(f"Done. Calculating the explained variance in the test dataset....")
        for func_data_test_subject in tqdm(func_data_test):
            scores_test.append(dictionary_learn.score(func_data_test_subject, confounds=None, per_component=False))
            scores_components_test.append(dictionary_learn.score(func_data_test_subject, confounds=None, per_component=True))
        scores_test_mean = np.mean(scores_test)
        scores_test_std = np.std(scores_test)
        scores_components_test = np.array(scores_components_test)
        scores_components_test_mean = np.mean(scores_components_test, axis=0)
        scores_components_test_std = np.std(scores_components_test, axis=0)

        scores_test = {
            "scores_test": scores_test,
            "scores_test_mean": scores_test_mean,
            "scores_test_std": scores_test_std,
            "scores_test_components": scores_components_test,
            "scores_test_components_mean": scores_components_test_mean,
            "scores_test_components_std": scores_components_test_std,
        }
        with open(dictionary_learn_scores_test_path, 'wb') as handle:
            pickle.dump(scores_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dual regression analysis
    for sub in func_data_train+func_data_test:
        fname = os.path.basename(sub.get_filename())
        sub_path_img = os.path.join(dictionary_learn_projections_path, "dr_"+fname)
        sub_path_corr = os.path.join(dictionary_learn_projections_path, "dr_"+fname.replace(".nii.gz", "_correlations.pickle"))
        if os.path.exists(sub_path_img) and os.path.exists(sub_path_corr):
            logging.info("Omitting already calculated dual_regression")
        else:
            logging.info(f"Calculating the dual regression of subject {fname}...")
            comp_img, correlations = dual_regression(sub, dictionary_learn, mask_file)
            comp_img.to_filename(sub_path_img)
            with open(sub_path_corr, 'wb') as handle:
                pickle.dump(correlations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return dictionary_learn