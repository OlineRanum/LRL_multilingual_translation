# Download and move data into right folder

wget http://phontron.com/data/ted_talks.tar.gz 
tar -xzvf ted_talks.tar.gz
rm ted_talks.tar.gz
mkdir src/preprocess/raw_ted_data
mkdir src/preprocess/split_data
mkdir src/preprocess/preprocessed_data
mv all_talks_test.tsv src/preprocess/raw_ted_data
mv all_talks_train.tsv src/preprocess/raw_ted_data
mv all_talks_dev.tsv src/preprocess/raw_ted_data
