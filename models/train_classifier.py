# import libs
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.externals import joblib
import pandas as pd
from sqlalchemy import create_engine
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Loads data from the database
    
    Args:
        database_filepath : filepath to the databse
        
    Return:
        X: 'message' column 
        Y: one-hot encoded categories
        category_names: category names included in Y
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM message_table;', con = engine)

    # splitting the target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    ''' 
    Tokenizes the text into words, nomalizes it and performs lemmatization
    
    Args: 
        text: Raw Text
    Return:
        clean_tokens: Lemmatized tokens containing only alphanumeric characters 
    '''
    #Removing non-alphanumeric characters
    text = re.sub('\W', '', text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # applying lematization
    clean_tokens = [lemmatizer.lemmatize(tok, 'n').lower().strip() for tok in tokens]
    clean_tokens = [lemmatizer.lemmatize(tok, 'v').lower().strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    '''
    Builds the entire pipeline
    
    Args:
        None
    Return:
        pipleline_ada : pipeline object
    '''
    
    #Defining the pipeline
    pipeline_ada = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer(use_idf = True)),
    ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators = 70, learning_rate = 1)))
    ])
    
    return pipeline_ada


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Evaluates the model, prints accuracy and the classification report
    
    Args:
        model: pipeline object
        X_test: Test features
        Y_test: Test labels
        category_names: names of categories present in the dataset
        
    '''
    
    Y_pred = model.predict(X_test)
    print("Accuracy:", (Y_pred == Y_test).sum()/Y_test.shape[0])
    
    #Classification Report
    Y_test = pd.DataFrame(Y_test)
    Y_pred = pd.DataFrame(Y_pred)
    for i in range(len(Y_pred.columns)):
        print(classification_report(Y_test.iloc[:,i].tolist(), Y_pred.iloc[:,i].tolist()))


def save_model(model, model_filepath):
    ''' 
    Saves the model for later use
    
    Args:
        model: Pipeline object
        model_filepath: model name
        
    return:
        None
    '''
    joblib.dump(model, open(model_filepath, 'wb'))


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