import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    "Reads csv files based on given file paths"

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner')

    return df

def clean_data(df):
    """Splits categories columns data based on ';'.
       Creates new columns for each splitted point.
       Assigns new column names.
       Removes duplicates
    """
    #Split
    categories = df["categories"].str.split(';',expand=True)

    #Extract new column names
    row = categories.iloc[:1]
    col_list = []
    for i in range(36):
        col_list.append(row[i][0][:-2])

    #Assign new column names
    categories.columns = col_list


    for col_name in col_list:

    # set each value to be the last character of the string
        categories[col_name]=categories[col_name].apply(lambda x: str(x)[-1:])

    # convert column from string to numeric
        categories[col_name]=categories[col_name].astype(int)

    #Drop old categories column
    df.drop(['categories'], axis=1, inplace=True)

    #Concate new categories dataframe to main df
    df = pd.concat([df,categories],axis=1)

    #Remove duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('labeledTrainingData', engine, index=False)


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
