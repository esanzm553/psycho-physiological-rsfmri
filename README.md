This repository contains all the code necessary to reproduce our article "The default mode network and behavior: A model to analyse psycho-physiological interactions in resting state fMRI" (in process of revision)

# Installation

1. Clone the repository and move into it:
```{bash}
git clone https://github.com/esanzm553/psycho-physiological-rsfmri.git
cd psycho-physiological-rsfmri
```
2. Create a python environment using `python -m venv venv` (this will create the environment in the "venv" folder relative to your current path)
3. Install the requirements file using `pip install -r requirements.txt`

# Input data

All the input data is publicly available on the Mind-Brain-Body page from the Max Plank Institute. You will need the following data to replicate our results [Link to the website](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html):

- Preprocessed fMRI volumes: [Link to download](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_MRI.html)
- Behavioral variables: [Link to download](https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/Compressed_tar/Behavioural_Data_MPILMBB_LEMON.tar.gz)

# Execution

1. Create a folder (e.g., data) in your project and put all the downloaded data inside
2. Modify the .env file from the .env.example with your own paths to the data
3. Enter the environment with `source venv/bin/activate`
4. Run the execution with `python run_pipeline.py` or with nohup (recommended) using `nohup python run_pipeline.py &` to execute the script in the background (the execution will continue if you close the terminal)

# Outputs

This script can take several hours or even days to execute. You can follow the execution in the logfile created in your "venv" folder. Most of the processes that take a long time are cached in the CACHE_FOLDER specified in the .env file. In the results file you will find all the statistics used in the article, as well as the NIfTI volumes containing the CanICA and DictionaryLearn results