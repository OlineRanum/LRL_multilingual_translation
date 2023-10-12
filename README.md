# DLNLP_Project
Project on low-resource multilingual translation for DLNLP.

## Installation

Installing the environment compatible with all components of the project and clone relevant repos (linux/ubuntu):

``` Installing and configuring repo
git clone https://github.com/OlineRanum/DLNLP_Project.git
cd DLNLP_Project

conda env create -f env.yml
conda activate dlnlp

bash setup_ubuntu_gpu.sh
```

For mac:
```
git clone https://github.com/OlineRanum/DLNLP_Project.git
cd DLNLP_Project

conda env create -f env.yml
conda activate dlnlp

cd src
cd langrank
get_data.sh
get_data.sh
python3 preprocess_datasets.py (uncomment  # preprocess_datasets())
precompute_features.py
```

## Preprocessing

After setting up the environment, data can be preprocessed. Code for this is in the ```src/preprocess``` folder. 

Preprocessing MUST be ran from main directory. Languages are specified using their lowercase two-letter code from all_talks.tsv (English -> en, Belarussian -> be, etc.) 

NOTE: All yaml files have an experiment name:
```
!Experiment
  name: preproc_eo_de_epoch100_multilingual
``` 
These must be changed each time the file is called, it will not start otherwise.

#### **Create Belarussian to English corpus with 100k max sentences for train, dev and test set respectively:**
``` 
python src/preprocess/ted_reader.py be 100000 100000 100000
```

#### **Preprocess corpus (tokenization, normalization, length filtering):**
```py
# Specify source and target language through SRC_LAN and TAR_LAN in preprocessing.yaml
xnmt src/preprocess/preprocessing.yaml
```

## Training

### Training bilingual models
To be able to train a bilingual model, ```src/preprocess/preprocessed_data``` must contain one folder named \<target language>_<source language\>, which was created by ```ted_reader.py```:

```
preprocess
|
└---preprocessed_data
|    |   en_be
    ...
...
```

Source and target language must be specified in through SRC_LAN and TAR_LAN ```train_preproc.yaml```, same as in the preprocessing step.

#### **Train bilingual model on GPU through PyTorch**:
```py
# Specify source and target language through SRC_LAN and TAR_LAN in train_preproc.yaml
xnmt train_preproc.yaml --backend torch --gpu
```

### Training multilingual models
To be able to train a multilingual model with $n$ transfer languages, we will need to $n$ monolingual corpora. Thus, ```src/preprocess/preprocessed_data``` must contain $n$ folders with the same target language by using ```ted_reader.py``` for each language:

```
preprocess
|
└---preprocessed_data
|    |   en_be
|    |   en_hu
|    |   en_az
|    ...
...
```
Create a folder to store your multilingual corpus in:
```
mkdir src/preprocess/preprocessed_data/merged_files/
mkdir src/preprocess/preprocessed_data/merged_files/be_hu_az
```

Combine the corpora by editing and running the notebook:
```py
jupyter notebook src/preprocess/preprocessed_data/
```

```py
###########################################
### In build_multilingual_dataset.ipynb ###

# SET languages
languages = ['be', 'hu', 'fa']

# SET number of sentences extracted per language for training
n_points = [4500, 10000, 10000]

# SET number of sentences extracted per language for dev
n_points_dev = [450, 1000, 1000]
###########################################

```

Finally, specify the multilingual corpus in ```train_preproc_multilingual.yml``` by editing ```DATA_IN```, ```DATA_EV``` and ```DATA_OUT``` and train the model.

#### **Train multilingual model on GPU through PyTorch**:
```py
# Specify source and target language by editing SRC_LAN and TAR_LAN in this file. Also specify which corpus to use by editing DATA_IN, DATA_EV and where to save it by editing DATA_OUT.
xnmt train_preproc_multilingual.yaml --backend torch --gpu
```

## Example
We will train a ```be-hu-fa -> en``` model.

```py
# Load data 
bash get_data.sh

# Split the data into the desired source languages
conda activate dlnlp
python src/preprocess/ted_reader.py be 100000 100000 100000
python src/preprocess/ted_reader.py hu 100000 100000 100000
python src/preprocess/ted_reader.py fa 100000 100000 100000

# Preprocess the data to usable formats for training.
python src/preprocess/preprocessing.yaml # changed SRC_LAN to be
python src/preprocess/preprocessing.yaml # changed SRC_LAN to hu
python src/preprocess/preprocessing.yaml # changed SRC_LAN to fa

# Create folders to put combined corpora in
mkdir src/preprocess/preprocessed_data/merged_files/
mkdir src/preprocess/preprocessed_data/merged_files/be_hu_az

# Combine corpora by editing n_points and languages in the notebook
jupyter notebook src/preprocess

# Train multilingual model on GPU after editing SRC_LAN, TAR_LAN, DATA_IN, DATA_EV, and DATA_OUT
xnmt train_preproc_multilingual.yaml --backend torch --gpu
```