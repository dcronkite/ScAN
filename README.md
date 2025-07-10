# ScAN: Suicide Attempt and Ideation Events Dataset

Re-implementation of ScAN and ScANER for identifying suicide attempt and suicidal ideation events in clinical notes.

* Original Repository: https://github.com/bsinghpratap/ScAN 
* Source Paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC9958515/ 


## About

This package is designed to:
* Build the ScAN dataset (done)
* Fine-tune a RoBERTa model to identify suicide attempt/ideation events (todo)
* Use the fine-tuned model to predict one's own clinical text (todo)


## Prerequisites

### MIMIC-III NOTEEVENTS dataset

ScAN was built using the MIMIC-III clinical database's NOTEEVENTS dataset. While this dataset is requestable via [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (see instructions below), it cannot be shared on Github. The relevant annotations have been provided to build the model, but the original dataset must be accessed separately. To do this,

* Create an account on [PhysioNet](https://physionet.org/):
  * Go to `Account` > `Register`
* Complete CITI trainings for Data or Specimens-only Research and Conflict of Interest
  * Go to the [trainings section of settings](https://physionet.org/settings/training/)
  * Follow the [step-by-step instructions](https://physionet.org/about/citi-course/) on that page
  * After completing the trainings, upload your completion report to the [trainings page](https://physionet.org/settings/training/)
* Identify verification and reason for use.
  * Go to the [credentialing section](https://physionet.org/settings/credentialing/) of your PhysioNet settings page
  * Identify yourself and role, provide a reference, and then explain your reason for use of the MIMIC-III datasets (i.e., reference this tool, cite the paper, and explain what you're doing with it)
* Wait for the identification verification and trainings to be verified
* Download the dataset
  * Navigate to the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
  * Complete the data use agreement
  * Download the dataset `NOTEEVENTS.csv.gz`
  * Download the checksum file `SHA256SUMS.txt`
  * Compare checksum in file to that of the dataset to confirm that you have the right bits and bytes:
    * Windows (powershell): `Get-FileHash -Path NOTEEVENTS.csv.gz -Algorithm SHA256`
    * Linux: `sha256sum NOTEEVENTS.csv.gz`
    * Mac: `shasum -a 256 NOTEEVENTS.csv.gz`
* Unzip the dataset
  * On Windows:
    * A tool like [7-zip](https://www.7-zip.org/download.html)
    * `wsl gunzip NOTEEVENTS.csv.gz` if you have Windows Subsystem for Linux
  * Otherwise: `gzip -d NOTEEVENTS.csv.gz`

### Code Setup

* Get this code/repository
  * `Option 1` (recommended):
    * Download git
      * E.g., https://git-scm.com/downloads
    * Find a workspace directory (e.g., `C:\code`)
    * Open terminal in that directory (e.g., powershell, etc.) 
    * Run `git clone https://github.com/dcronkite/ScAN`
    * This will clone (i.e., 'copy') the repository to `C:\code\ScAN`
  * `Option 2`: Download zip archive from GitHub 
    * `Get Code` > `Download ZIP`
* Setup Python
  * Download Python 3.13+
    * https://www.python.org/downloads/
  * Setup a virtual environment
    * Open the cloned `ScAN` directory in a terminal 
    * `python -m venv .venv`
  * Install prerequisite packages/libraries (one of the following):
    * `pip install .`
    * OR with uv, `uv sync`

## Usage


