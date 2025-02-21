import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''  Loads the csv files from the given path
    Args:
        Messages_filepath : path to message.csv
        categories_filepath: path to categories.csv
    
    Return:
        Returns merged dataframed (merged_df) containing data from both 
        messages.csv and categories.csv
    '''
    #Reading dataframes messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merging both dataframes using the id
    df_merged = messages.merge(categories, how = 'inner', on =['id'] )
    
    return df_merged

def clean_data(df):
    ''' 
        Cleans the data of the merged dataframe
         - Renames columns of different categories
         - Drops duplicates

         Args:
            df : Merged dataframe containing data from messages.csv and categories.csv

         Return:
            df: Processed dataframe
    '''
    # Split `categories` into separate columns.
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # Renaming the column names using the first row
    row = categories.iloc[0,:].tolist()
    category_colnames = [col_name[:-2] for col_name in row ]
    categories.columns = category_colnames
    
    #replacing original values with 1 and 0 (encoding values)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda value: value[-1])
        categories[column] = pd.to_numeric(categories[column])

    #drop the categories column
    df.drop('categories', axis = 1 , inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    #drop duplicates
    df = df.drop_duplicates()
    
    return df
            
            
def save_data(df, database_filename):
    '''
    Saves the dataframe into the database
    
    Args:
        df: dataframe to be saved
        database_filename :filepath of the database file
        
    Return:
        None
    '''

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_table', engine, index=False, if_exists='replace')

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