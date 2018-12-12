import sys

# import libraries
import pandas as pd
import numpy as np

# Setting up connection to sqlite
import sqlite3
# Import NLTK libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Import necessary sklearn libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import re
# to save the model
from joblib import dump

nltk.download(['punkt', 'wordnet', 'stopwords'])

'''
Example of execution:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
'''

def load_data(database_filepath):
    '''
    Load the data from a sqlite database

    Args:
        database_filepath  : (relative) filepath of sqlite database

    Returns:
        Combined dataset removed from duplicate rows
    '''
    # Set up a connection to sqlite database
    conn = sqlite3.connect(database_filepath)
    # load data from database, the tablename is hardcoded in the script that performs the ETL.
    df = pd.read_sql('SELECT * FROM messages', conn)
    # The data needs to be 1 dimensional for countvectorizer and tfidftransformer
    X = df.message.tolist()
    # Get all values that we are trying to predict
    # Turns out that column child_alone has no value. So for this dataset it makes sense to remove this.
    y = df.drop(columns=['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    # Get all category names that we are trying to predict
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Tokenize the message

    Args:
        text : A message that needs preprocessing

    Returns:
        Removed punctation, lower cased all letters, removed stopwords, lemmatized and stemmed all words.
    '''
    # First remove punctation and lowercase all letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # lemmatize, stem and remove stop words
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    '''
    Creates a sklearn pipeline object

    Args:
        None

    Returns:
        A sklearn pipeline object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_leaf=1)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the performance of the model for all the different categories that we are trying to predict.

    Args:
        model          : The model that we will use to make the predictions with
        X_test         : Attributes that we will use to make the predictions
        Y_test         : True labels for the corresponding attributes in X_test
        category_names : list of categories that can be predictied

    Returns:
        None
    '''
    # Make a prediction
    Y_pred = model.predict(X_test)
    
    idx = -1
    for column in category_names:
        idx += 1
        print(column)
        print('_'*60)
        print(classification_report(Y_test[column], Y_pred[:,idx]))
        print("Accuracy: {0:.4f}".format(accuracy_score(Y_test[column], Y_pred[:,idx])))
        print("\n")


def save_model(model, model_filepath):
    '''
    Saves the state of the (trained) model

    Args:
        model          : The trained model
        model_filepath : (relative) path where to store the model

    Returns:
        None
    '''
    dump(model, model_filepath)


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