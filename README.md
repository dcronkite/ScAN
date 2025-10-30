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

### Models

Select a `RoBERTa-base` model for the fine-tuning. Note that the original medRoberta model was trained on internal data and therefore cannot be released (though you could always perform some preliminary training on your own data).

* RoBERTa-base: https://huggingface.co/FacebookAI/roberta-base


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

### Creating an Annotated MIMIC III Corpus

Prior to fine-tuning, we will need to add annotations to the MIMIC-III Corpus. This will create 3 jsonlines files (and an intermediate `files/` directory) which can then be supplied to the trainer:

**Output Structure:**
```
outdir/
├── files/                  # contains intermediate processed text files
├── train.annot.jsonl       # training set annotations
├── test.annot.jsonl        # test set annotations (no annotations?)
├── val.annot.jsonl         # validation set annotations
├── train.jsonl             # training set labels
├── test.jsonl              # test set labels (no annotations?)
└── val.jsonl               # validation set labels
```

**Parameters:**
- `note_events`: oath to MIMIC-III NOTEEVENTS.csv file
- `outdir`: output directory for the ScAN corpus
- `-s, --chunk-size`: number of sentences in each chunk (default: 20)
- `-o, --overlap`: number of overlapping sentences between chunks (default: 5)

**Example Command:**
```bash
scan-build /data/NOTEEVENTS.csv ./scan_corpus --chunk-size 20 --overlap 5
```

### Training the Classifier (evidence retriever)

This script trains a RoBERTa-based model to identify suicide attempt and ideation events using the annotated corpus created in the previous step.

**Parameters:**
- `--config`: Path to configuration YAML file (required)
  - See example [train_config.yaml](examples/train_config.yaml) and update path to `model.name`
- `--train-path`: Path to `train.jsonl` generated from `create_scan_corpus.py` (required)
- `--output-dir`: Output directory for model checkpoints (default: ./checkpoints)
- `--gpu-id`: GPU ID to use for training (default is `0` for cpu-only)
- `--test-path`: Path to `test.jsonl` generated from `create_scan_corpus.py` (optional)
- `--val-path`: Path to `val.jsonl` generated from `create_scan_corpus.py` (optional)

**Example Command:**
```bash
scan-train --config train_config.yaml \
      --train-path ./scan_corpus/train.jsonl \
      --val-path ./scan_corpus/val.jsonl \
      --test-path ./scan_corpus/test.jsonl
```
The trained model will be saved to the output directory specified at the bottom of the configuration file under `paths.output_dir` with the name specified in `paths.model_name`.
