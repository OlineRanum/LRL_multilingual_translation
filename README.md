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