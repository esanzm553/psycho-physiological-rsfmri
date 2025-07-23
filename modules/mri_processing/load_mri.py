import nibabel as nib
from glob import glob
import os
import pickle
import re
import pandas as pd

labelmaps = {
    "canica": {
        0: "Occipital Pole 1",
        1: "Temporal Pole",
        2: "Sensorimotor 1",
        3: "Noise",
        4: "Noise",
        5: "Sensorimotor 2",
        6: "Noise",
        7: "Lateral Visual 1",
        8: "Noise",
        9: "Premotor",
        10: "Rostral DMN",
        11: "Parietal",
        12: "Noise",
        13: "Occipital Pole 2",
        14: "DMN Dorsal Lateral",
        15: "Noise",
        16: "Frontoparietal Right",
        17: "Caudal DMN",
        18: "Sensorimotor 3",
        19: "Noise",
        20: "Cerebellum",
        21: "Noise",
        22: "Prefrontal Dorsolateral",
        23: "Noise",
        24: "Frontoparietal Left",
        25: "Auditory",
        26: "Noise",
        27: "Noise",
        28: "Lateral Visual 2",
        29: "Sensorimotor 4"
    },
    "dictionary_learn": {
        0: "Sensorimotor 1",
        1: "Noise",
        2: "Noise",
        3: "Frontoparietal Right",
        4: "Noise",
        5: "Noise",
        6: "Noise",
        7: "Sensorimotor 2",
        8: "Frontoparietal Left",
        9: "Sensorimotor 3",
        10: "Medial Motor",
        11: "Occipital Pole 1",
        12: "Noise",
        13: "Ventromedial",
        14: "Occipital Pole 2",
        15: "Auditory",
        16: "Noise",
        17: "Sensorimotor 4",
        18: "Noise",
        19: "DMN",
        20: "Auditory + Frontoparietal",
        21: "Lateral Visual",
        22: "DMN Dorsal",
        23: "Noise",
        24: "Anterior Parietal",
        25: "Middle Parietal",
        26: "Frontopatietal Right",
        27: "Frontoparietal Bilateral Middle",
        28: "Frontoparietal Bilateral Superior",
        29: "Cerebellum"
    }
}

def mri_data_loader(folder:str, file_pattern:str="*") -> list:
    files = glob(f"{folder}/**/{file_pattern}*.nii*", recursive=True)
    nib_files = [nib.load(f) for f in files]
    return nib_files

def check_data(data_list:list):
    error_files = []
    for f in data_list:
        try:
            f.get_fdata()
        except:
            fname = f.get_filename()
            print(f"Error processing {fname}")
            error_files.append(fname)
        f.uncache()
    return error_files

def read_dual_regression_data(mri_df:pd.DataFrame, dr_data_path:str, prefix:str):
    noise_comps = [i for i,x in labelmaps[prefix].items() if x == "Noise"] if prefix in labelmaps.keys() else []
    for f in os.listdir(dr_data_path):
        if f.endswith(".pickle"):
            with open(os.path.join(dr_data_path,f), "rb") as handle:
                dr_data = pickle.load(handle)
            sub = [y for y in mri_df.index if re.search(y, f)]
            for i,c in enumerate(dr_data):
                if i not in noise_comps:
                    mri_df.loc[sub, "_".join([prefix, str(i)])] = c
    return mri_df