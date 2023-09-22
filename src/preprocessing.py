import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DFPreprocessor:
    def __init__(self):
        # Initialize parameters needed for text preprocessing
        #self.stopwords = set(stopwords.words('english'))
        #self.stemmer = nltk.PorterStemmer()  

        self.languages = ['en', 'be']
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

        # Remove all null pairs
        df = self.apply_remove_null_columns(df)

        # Write to output file
        self.write_df_to_headfile(df)
        self.extract_vocabulary_from_dataframe(df)
        
        return df
    
    def apply_lowercase(self, df):
        return df.applymap(lambda doc: doc.lower() if type(doc) == str else doc)
    
    def apply_remove_characters(self, df):
        for char in self.remove_char:
            df = df.applymap(lambda s: s.replace(char, '') if type(s) == str else s)
        return df
    
    def apply_remove_null_columns(self, df, value_to_remove = ['__null__', '_ _ null _ _']):
            """ NB! TODO: This is not optimal: some documents contain singular instances of __null__ marker to indicate that a 
            word is missing. Others docs are comprised of several instances of _ _ null _ _ __null__.
            Need to test what to do with this
            """
            for index, row in df.iterrows():
                # Check if the row contains the value to remove in any column
                for value in value_to_remove:
                    if any(row == value):
                        # If it contains the value, remove the entire row
                        df.drop(index, inplace=True)
            return df

    
    def extract_vocabulary_from_dataframe(self, df, output_dir = '../Data/Preprocessed'):
        """
        Write each column of a DataFrame to separate vocabulary files compatible with the xnmt framework

        Parameters:
        df (df): The DataFrame to extract data from.
        output_directory (str): The directory where the text files will be saved.

        Output:
        file (txt): comprised of vocabulary extracted from each column/language.
        """
        # Create a dictionary to store vocabularies for each language
        vocabularies = {}
        
        # Iterate over columns (languages) in the DataFrame
        for column in df.columns:
            # Initialize an empty set to store unique words
            word_set = set()
            
            for text in df[column]:
                # Tokenize the text into words
                words = word_tokenize(text)
                
                # Add the words to the set to ensure uniqueness
                word_set.update(words)
            
            vocabularies[column] = word_set
            
            # Write the vocabulary to a text file
            output_file = f"{output_dir}/{column}.vocab.txt"
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write('\n'.join(word_set))
        
        return vocabularies
        
    def write_df_to_headfile(self, df, output_directory  = '../Data/Preprocessed'):
        """
        Write each column of a DataFrame to separate text files compatible with the xnmt framework

        Parameters:
        df (df): The DataFrame to extract data from.
        output_directory (str): The directory where the text files will be saved.

        Output:
        file (txt): One text file per column, with each row in the text file representing a cell from that column.
        """
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(output_directory, exist_ok=True)
        
        # Loop through each column in the DataFrame and create txt file
        for column_name, column_data in df.items():
            output_file_path = os.path.join(output_directory, f"head.{column_name}.txt")
            
            # Write the column data to the text file
            with open(output_file_path, "w") as text_file:
                text_file.write("\n".join(map(str, column_data)))
                
            print(f"Saved '{column_name}' to {output_file_path}")



    def preprocess_dataframe(self, path):
        """
        Preprocess a dataframe comprised of parallel documents.
        
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
