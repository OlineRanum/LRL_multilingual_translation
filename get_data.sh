# Download and move data into right folder

wget http://phontron.com/data/ted_talks.tar.gz 
tar -xzvf ted_talks.tar.gz
rm ted_talks.tar.gz
mv all_talks_test.tsv src/preprocess/raw_ted_data
mv all_talks_train.tsv src/preprocess/raw_ted_data
mv all_talks_dev.tsv src/preprocess/raw_ted_data