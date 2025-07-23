from modules.mri_processing.load_mri import mri_data_loader
from modules.mri_processing.canICA import run_canICA
from modules.mri_processing.dictionaryLearn import run_dictionary_learn
from modules.mri_processing.preprocess_mri import compute_mask, compute_mean_images
from modules.covariates_processing.preprocess_covariates import load_and_preprocess_covariates, load_and_preprocess_metadata, join_mri_covariates
from modules.covariates_processing.analyze_covariates import optimize_and_reduce_covariates, calculate_feature_importances
from modules.modelling.univariate_analysis import pipeline_univariate_analysis
from modules.modelling.multivariate_analysis import pipeline_multivariate_analysis
import os
from random import shuffle, seed
import pickle
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def run_pipeline(folder:str, personality_test_folder, cognitive_test_folder, metadata_file, 
                 func_pattern:str, output_path:str, split_train:float=0.7, 
                 n_jobs:int=2, cache:str|None=None, graphics_path="./"):
    ## PROCESSING OF MRI
    # Load fMRI data
    logging.info("Loading fMRI data")
    seed(777)
    func_data = mri_data_loader(folder, func_pattern)
    shuffle(func_data)
    thr = int(len(func_data)*split_train)
    func_data_train = func_data[:thr]
    func_data_test = func_data[thr:]
    logging.info(f"Loaded {len(func_data)}. Train: {len(func_data_train)}. Test: {len(func_data_test)}")
    # Calculate mean imgs to make the mask
    mean_imgs_folder = os.path.join(output_path, "mean_imgs")
    os.makedirs(mean_imgs_folder, exist_ok=True)
    compute_mean_images(func_data_train, mean_imgs_folder)
    mean_imgs = mri_data_loader(mean_imgs_folder, "mean_img_*")
    #Calculate the mask
    mask_file = compute_mask(mean_imgs, output_path, cache)
    # State of the art analysis: CANICA
    canica_output_path = os.path.join(output_path, "canica")
    canica_cache = os.path.join(cache, "canica")
    canica = run_canICA(canica_output_path, func_data_train, func_data_test, n_components=30, n_jobs=n_jobs, cache=canica_cache, mask_file=mask_file)
    # State of the art analysis: Dictionary Learn
    dictionary_learn_output_path = os.path.join(output_path, "dictionary_learn")
    dictionary_learn_cache = os.path.join(cache, "dictionary_learn")
    dictionary_learn = run_dictionary_learn(dictionary_learn_output_path, func_data_train, func_data_test, n_components=30, n_jobs=n_jobs, cache=dictionary_learn_cache, mask_file=mask_file)

    ## COVARIATES PROCESSING
    covariates_output_path = os.path.join(output_path, "covariates")
    os.makedirs(covariates_output_path, exist_ok=True)
    all_data_path = os.path.join(covariates_output_path, "all_data_joined.xlsx")
    embedding_model_path = os.path.join(covariates_output_path, "embedding_model.pickle")
    if os.path.exists(all_data_path) and os.path.exists(embedding_model_path):
        logging.info("Loading previously calculated covariates")
        all_data_df = pd.read_excel(all_data_path)
        with open(embedding_model_path, "rb") as handle:
            embedding_model = pickle.load(handle)
    else:
        # Load the covariates
        logging.info("Loading covariates")
        covariates = load_and_preprocess_covariates(cognitive_test_folder, personality_test_folder)
        metadata = load_and_preprocess_metadata(metadata_file)
        # Reducing the dimensionality of the phenotipic and cognitive data
        logging.info("Reducing the dimensionality of the covariates")
        embedding_model, covariates_reduced = optimize_and_reduce_covariates(covariates, graphics_path, covariates_output_path)
        with open(embedding_model_path, "wb") as ref:
            pickle.dump(embedding_model, ref, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Calculating the importance of each variable in the model")
        calculate_feature_importances(covariates, covariates_reduced, graphics_path, covariates_output_path)
        # Join all the results
        all_data_df = join_mri_covariates(func_data_train, func_data_test, canica_output_path, dictionary_learn_output_path, covariates_reduced, metadata)
        all_data_df.to_excel(all_data_path)
    
    ## DATA MODELLING
    pipeline_multivariate_analysis(all_data_path, output_path)
    pipeline_univariate_analysis(all_data_path, output_path)
    logging.info("Finish")


if __name__ == "__main__":
    images_folder = os.environ.get("IMAGES_FOLDER", "data/MRI_Preprocessed_Derivetives/MRI_Preprocessed_Derivetives")
    personality_test_folder = os.environ.get("PERSONALITY_TEST_FOLDER", "data/Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON")
    cognitive_test_folder = os.environ.get("COGNITIVE_TEST_FOLDER", "data/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON")
    metadata_file = os.environ.get("METADATA_FILE", "data/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")
    cache_folder = os.environ.get("CACHE_FOLDER", "venv/cache/")
    graphics_path = os.environ.get("GRAPHICS_PATH", "venv/graphics")
    date_fmt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    logging.basicConfig(filename=f"venv/logfile_{date_fmt}.log", filemode="a", format='[%(asctime)s]: %(message)s', encoding='utf-8', level=logging.INFO)
    output_path = os.path.join("venv", "results")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(graphics_path, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)
    logging.info(f"Using the following folders:\n\t{output_path}\n\t{graphics_path}\n\t{cache_folder}")

    run_pipeline(folder=images_folder,
                 personality_test_folder=personality_test_folder,
                 cognitive_test_folder=cognitive_test_folder,
                 metadata_file=metadata_file,
                 func_pattern="*task-rest*MNI2mm",
                 anat_pattern="*",
                 output_path=output_path,
                 n_jobs=4,
                 cache=cache_folder,
                 graphics_path=graphics_path)
