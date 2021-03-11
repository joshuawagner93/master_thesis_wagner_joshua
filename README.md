# master_thesis_wagner_joshua

First installation:
 - with pip: pip install -r requirements.txt
 - with conda: either:
    - conda create --name <env_name> --file conda_requirements.txt
    - conda env create -f environment.yml
 - the conda create from environment.yml file contains the env name, in my case the base environment, change this to any name you want so it won't mess with your base env.

!!! Original Datasets are zipped for upload !!! 

Datasets in ./datasets/m4/train , ./datasets/m4/test and ./datasets/uci_har/v2 will need to be unzipped before any code requiring the original data can be run.
This includes models if the saved preprocessed data is not available. The different models will use the same preprocessed data if it is available in ./code/**/processed_data .


All paths in the code in the ./code/ folder are relativ to that folder.
The code is run using the Spyder IDE and uses a spyder project in ./code/ so all code is run with that folder as its origin.
If no new spyder project is initialized, an os.chdir() command will be necessary before running any other code.

Results are also contained in the corresponding ./code/**/results/ folder.
Aggregated results can be found in the ./code/aggregated_results/ folder for either M4 or HAR.
