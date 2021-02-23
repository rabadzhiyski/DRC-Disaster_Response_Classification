import sys
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages.csv, categories.csv):
    """Load data files by providing the file name or path
    """
    messages = pd.read_csv('messages.csv')
   
    categories = pd.read_csv('categories.csv')
    

def clean_data(df):
    """Merge data sets, split categories into separate columns,
    convert category values to just numbers 0 and 1, replace
    categories column in df with new category columns
    """
    #merge data
    df = messages.merge(categories, how="inner", on=['id'])


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
