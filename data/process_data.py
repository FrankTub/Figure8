import sys
import pandas as pd
import numpy as np

# Setting up connection to sqlite
import sqlite3

'''
Example of execution:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
'''


def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from two csv files

    Args:
        messages_filepath  : (relative) filepath of messages.csv
        categories_filepath: (relative) filepath of categories.csv

    Returns:
        Combined dataset removed from duplicate rows
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv('./data/categories.csv')
    
    # Remove exact duplicate rows from the datasets
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
    
    # Remove double id columns in categories
    categories.drop_duplicates(subset='id', keep="last", inplace=True)
    # Merge the datasets
    df = pd.merge(messages, categories, on='id', how='inner')
    return df

def clean_data(df):
    '''
    Clean the category column and make columns for all values

    Args:
        df: Pandas dataframe

    Returns:
        Cleaned category dataset
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe and assume that all different 
    # values exist in this row
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str[:-2])
    # Change the column names
    categories.rename(columns=category_colnames.iloc[0], inplace=True)
    
    for column in categories:
    # set each value to be the last character of the string
    # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x[-1:]))
    
    # drop the original categories column from `df`
    df.drop(columns='categories', axis=0, inplace=True)
    
    # Combine the dataframes and return this
    df = pd.concat([df, categories], axis=1)
    return df


def save_data(df, database_filename):
    '''
    Store the dataframe in a sqlite database on your local machine

    Args:
        df               : Pandas dataframe
        database_filename: (relative) filepath of sqlite database

    Returns:
        None
    '''
    conn = sqlite3.connect(database_filename) 
    df.to_sql("messages", conn, if_exists="replace", index=False)    


def main():
    '''
    Function that will be called by executing this script
    "python process_data.py"
    '''
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