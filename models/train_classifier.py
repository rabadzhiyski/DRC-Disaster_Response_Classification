# import libraries
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet'])

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, confusion_matrix

# define functions
def load_data(database_filepath):
    """
    Load the data from the database
    """
    # create engine
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql("DisasterResponse", engine)
    X = df['message']
    
    #drop also "related" column as there were records with 3 choice
    Y = df.drop(['id', 'message', 'original', 'genre', 'related'], axis = 1)
    
    # add category names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and clean text.
    
    Args: message text
    
    Returns: normalized tokens
    """
    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build ML pipeline and use Grid to find the best parameters
    """
    # pipeline model
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, oob_score=True, 
                                                         n_jobs=-1,random_state=50, max_features="auto", 
                                                         min_samples_leaf=50)))
    ])
    # use grid to tune model 
    parameters = {'tfidf__norm': ['l1','l2'],
              'clf__estimator__criterion': ["gini", "entropy"]
    
             }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Asses model's performance with confusion matrix'
    """
    # make predictions
    Y_pred = model.predict(X_test)
    
    # print classificatin report
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Build a picle file and save model
    """
    pickle.dump(model, open(model_filepath,'wb'))


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