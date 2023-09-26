
# Ensure cmake is available for compiling dynet
pip install cmake
## Install Lightgmb
conda install -c conda-forge lightgbm

# Lang2Vec installation
git clone https://github.com/antonisa/lang2vec.git
cd lang2vec
wget http://www.cs.cmu.edu/~aanastas/files/distances.zip .
mv distances.zip lang2vec/data/
python3 setup.py install
cd ../

# Langrank installation
git clone https://github.com/neulab/langrank.git
cd langrank 
pip install -r requirements.txt
wget http://phontron.com/data/langrank/indexed.tar.gz  .
tar -xzvf indexed.tar.gz
rm indexed.tar.gz
cd ../

# XNMT
git clone https://github.com/neulab/xnmt.git
cd xnmt
pip install -r requirements.txt
python setup.py install

# Install correct version of protobuf
pip uninstall protobuf
pip install protobuf==3.20.0

# Get data
mkdir src/preprocess/raw_ted_data
mkdir src/preprocess/split_data
mkdir src/preprocess/preprocessed_data

wget http://phontron.com/data/ted_talks.tar.gz 
tar -xzvf ted_talks.tar.gz
rm ted_talks.tar.gz
mv all_talks_test.tsv src/preprocess/raw_ted_data
mv all_talks_train.tsv src/preprocess/raw_ted_data
mv all_talks_dev.tsv src/preprocess/raw_ted_data

# END
python << END
print('------- Set-up for DLNLP-project complete -------\n')
END
