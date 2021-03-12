# master_thesis_wagner_joshua

## First installation:
 - with pip: pip install -r requirements.txt
 - with conda: either:
    - conda create --name <env_name> --file conda_requirements.txt
    - conda env create -f environment.yml
 - the conda create from environment.yml file contains the env name, in my case the base environment, change this to any name you want so it won't mess with your base env.

### !!! Original Datasets are zipped for upload !!! 

Datasets in ./datasets/m4/train , ./datasets/m4/test and ./datasets/uci_har/v2 will need to be unzipped before any code requiring the original data can be run.
This includes models if the saved preprocessed data is not available. The different models will use the same preprocessed data if it is available in ./code/**/processed_data .


All paths in the code in the ./code/ folder are relativ to that folder.
The code is run using the Spyder IDE and uses a spyder project in ./code/ so all code is run with that folder as its origin.
If no new spyder project is initialized, an os.chdir() command will be necessary before running any other code.

Results are also contained in the corresponding ./code/**/results/ folder.
Aggregated results can be found in the ./code/aggregated_results/ folder for either M4 or HAR.

### HAR scripts:

  - har_test_script.py : contains code snippets used in the functions, mainly used for tests
  
  - har_trials.py : script with a cwt trail run unpacked from the looping functions to test where errors could exist
  
  - har_v2_f.py : script containing the functions to run variations of the har analysis with different preprocessings and models. This file only contains functions and is not meant to be run by itself.
  
  - check_uci_har_cnn_results : contains a check function if the saved results can be replicated with the corresponding model and preprocessed dataset. Also contains the loop to run over the different parameters in the preprocessing. Careful with these models as the preprocessing and training take about ~1.5h per parameter and model. Possible parameters are in comments above the variable declaration.
  
  - uci_har_dwt_models.py : contains the loop for the models using the raw time-series or the DWT coefficients version. Possible parameters are again in comments above the variable declaration. The loops may need adjusting according to the chosen parameters.
  
  - uci_har_feat_models.py : contains the loops for the feature models and the models trained with the original features. The feature loop needs adjustment based on which parameters are varied as all variations in one run would take to long.

  

### M4 scripts:

  - m3_trials.py : miss-named toy file containing first experiments with the m4 dataset.
  
  - m4_toy_file.py : as the name implies a toy file containing various code snippets and functions that test the functions in the actual variable function.
  
  - m4_f.py : contains all the functions used to run variable preprocessings on the m4 dataset.
  
  - run_m4_f.py : does what the name says, contains the loops for the m4 functions with variable parameters. Contains loops for both multires. and denoising variations. 


### Misc. scripts:

  - aggregate_har_results.py : used to aggregate all the various csv files containing the results of the models. Contains modifications to variable names for the latex tables. Used for the boxplots and tables in the thesis.
  
  - aggregate_m4_results.py : same as the har results just smaller as there are less variations for the m4 data.
  
  - confusion_matrices.py : contains functions and runs them to create the confusion matrices. Needs the models and the saved preprocessed data to be available.
  
  - frequency_plots.py : trials with frequency plots of the example signal in the thesis. 
  
  - wavelets_images.py : used to create the discrete wavelet images in the appendix of the thesis.