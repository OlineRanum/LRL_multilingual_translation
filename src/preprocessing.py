import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DFPreprocessor:
    def __init__(self):
        # Initialize parameters needed for text preprocessing
        #self.stopwords = set(stopwords.words('english'))
        #self.stemmer = nltk.PorterStemmer()  

        self.languages = ['en', 'es']
        #self.languages = ['talk_name'] + self.languages

        # Characters to remove from text
        self.remove_char = ['.', ',']


    def clean_df(self, df):
        """
        Clean and preprocess a dataframe comprised of multiple parallel documents.
        
        Parameters:
        df (df): The input dataframe to preprocess.
        
        Returns:
        df (df): The preprocessed dataframe.
        """
        # Convert df to lowercase
        df = self.apply_lowercase(df)
        
        # Remove punctuation and other special characters
        df = self.apply_remove_characters(df)
        
        return df
    
    def apply_lowercase(self, df):
        return df.applymap(lambda doc: doc.lower() if type(doc) == str else doc)
    
    def apply_remove_characters(self, df):
        for char in self.remove_char:
            df = df.applymap(lambda s: s.replace(char, '') if type(s) == str else s)
        return df


    def preprocess_dataframe(self, path):
        """
        Preprocess a dataframe comprised of parallel corpa.
        
        Parameters:
            path (str): path to df of parallel documents to preprocess.
            
        Returns:
        preprocessed df (df): The preprocessed parallel documents.
        """
        # Make new dataframe comprised of relevant languages
        dev = pd.read_csv(path, sep = '\t', usecols=self.languages)
        
        # Clean dataframe
        preprocessed_df = self.clean_df(dev)
        
        return preprocessed_df
    


if __name__ == "__main__":
    
    tsv_path = "../Data/all_talks_dev.tsv"  
    text_preprocessor = DFPreprocessor()
    df = text_preprocessor.preprocess_dataframe(tsv_path)
