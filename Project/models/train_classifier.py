import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    'Reads clean data from sql database returns as pandas dataframe'
    #create sql engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    #read data from sql database
    df = pd.read_sql("Select * from labeledTrainingData",engine)

    #Messages
    X = df.iloc[:,1]

    #Categories
    y = df.iloc[:,4:]

    #extract category names
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    Removes punctions and lowercases sentences.
    Tokenizes sentences using nltk library.
    Removes stopwords.
    Uses lemmatizer.
    Returns normalized, tokenized, lemmatized list of words.
    """
    #Punctiation removal and lowercase
    text = re.sub(r"[^a-zA-Z0-9]"," ", text).lower()

    #Tokenization
    words = word_tokenize(text)

    #Stop word removal
    words = [w for w in words if w not in stopwords.words("english")]

    #Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word.strip())
        clean_words.append(clean_word)

    return clean_words


def build_model():

    #Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #Define your parameters for gridsearch
    parameters = {'clf__estimator__min_samples_split':[4,8,16],
             }

    #Create GridSearchCV object
    grid_obj = GridSearchCV(pipeline,param_grid=parameters)

    return grid_obj


def evaluate_model(model, X_test, Y_test, category_names):

    #Use model to make prediction on testing data
    y_pred = model.predict(X_test)

    #Print out classification report for every class
    for i in range(36):
        print(category_names[0], classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
