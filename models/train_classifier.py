import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    # Load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)

    columns = df.columns[4:]
    X = df['message'].values
    Y = df[df.columns[4:]].values
    
    return X, Y, columns


def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text (split text into words)
    tokens = word_tokenize(text)
    
    # Remove stop words
    words = [word for word in tokens if word not in stopwords.words('english')]

    # Lemmatization (reduce words to their root form)
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    # Build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight='balanced'))
    ])
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pred = model.predict(X_test)

    precisions, recalls, f1s = [], [], []
    for idx,column in enumerate(category_names):
        precisions.append(precision_score(Y_test[:,idx], pred[:,idx]))
        recalls.append(recall_score(Y_test[:,idx], pred[:,idx]))
        f1s.append(f1_score(Y_test[:,idx], pred[:,idx]))

    report = pd.DataFrame({'Categories': category_names, 'Precision': precisions, 'Recall': recalls, 'F1': f1s})
    print(report)


def save_model(model, model_filepath):
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


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