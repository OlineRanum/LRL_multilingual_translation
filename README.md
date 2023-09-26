# DLNLP_Project
Project on multilingual translation for DLNLP

## Installation

For installing environment compatible with all components of the project and clone relevant repos (linux/ubuntu)

``` Installing and configuring repo
git clone https://github.com/OlineRanum/DLNLP_Project.git
cd DLNLP_Project

conda env create -f env.yml
conda activate dlnlp

bash setup_ubuntu_gpu.sh
```


## Get Data

Download data to correct folder (I've also added this to the setup file, but for your convenience)

```
bash get_data.sh
```


## Run code 

### Preprocess data

NB! Language pairs is set in ted_reader and in all yaml files. 
All commands run from main directory. 

#### Split raw data into train-dev-test sets
```
python3 src/preprocess/ted_reader.py
```
#### Preprocess data
```
xnmt src/preprocess/preprocessing.yaml
```

#### Train models
```
xnmt train_preproc.yaml
```
or
```
xnmt train.yaml
```

