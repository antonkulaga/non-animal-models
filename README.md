# non-animal-models

This code is prototype of the extraction from non-animal models datasets.
It is super-early and produces bad early results so far.

# Progress update

What has been done so far:
* Micromamba environment with all required dependencies
* DVC management of the datasets (all datasets are downloaded and DVC-controled)
* Downloading most of the papers in existing datasets - 2585 papers has been downloaded so far
* Parsing of the downloaded papers with 1613 papers has been successfully parsed, for others some issues can be fixed.
*  Setting up classes to work with NAM datasets
* Converting papers to OpenAI embeddings
* Applying langchain prompts with ChatGPT API for classifications with configurable fields and clarifications
* Tracking results with https://langchain.plus/
## Datasets used in the prototype:

* [Respiratory diseases](https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/Respiratory_data%20catalogue_v2September2020.xlsx)
* [Neurogenerative diseases](https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/Neurodegenerative_data%20catalogue_v27April2021.xlsx)
* [Breast cancer](https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/Breast%20cancer_data%20catalogue.xlsx)
* [Immuno-oncology](https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/Immuno-oncology_data%20catalogue.xlsx)
* [Immunogenicity](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/ATMP_data_catalogue_v23March2022.xlsx)
* [Autoimmunity](https://data.jrc.ec.europa.eu/dataset/700397b2-edd7-4ed6-86f7-fc1b164ed432)
* [Cardio-vascular diseases](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/BiomedResearchReview/Carciovascular_data_catalogue_v29September2022.xlsx)


# Getting started

To create environment use micromamba:
```bash
micromamba create -f environment.yaml
micromamba activate non-animal-models
```

## Pull data from DVC
We are using [DVC](http://dvc.org) for data version control.
It is required because database index does not fit into git.
To set it up do:
```bash
dvc pull
```
We use Google Drive to store data, so it may ask you to authorize data access.

# Environment

To run openai API and langchain tracing you need keys in your environment variables.
Do not forget to put on your openai and langchain trace keys.
In .env.template there are environment variables, fill them in with your keys and rename to .env.


# Preprocessing
Our chains use two indexes one for papers and one for database.
Overall you can just pull the data from DVC (it may ask to confirm google drive access)
```bash
micromamba activate non-animal-models
dvc pull
```
However, if you want to update indexes you can use preprocessing scripts at
preprocess.py
After the update you update data of the DVC repo by:
```bash
dvc commit
dvc push
```

Overall, most of the scripts required for preprocessing can be called from dvc, for example:
![Alt text](./dvc_pipeline.svg)