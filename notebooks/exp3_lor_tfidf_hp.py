import mlflow
import dagshub
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re 
import string
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

mlflow.set_tracking_uri("https://dagshub.com/nmathew23/mlops-mini-project.mlflow")
dagshub.init(repo_owner='nmathew23', repo_name='mlops-mini-project', mlflow=True)

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
df.head()

# data preprocessing

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

df = normalize_text(df)
df.head()

x = df['sentiment'].isin(['happiness','sadness'])
df = df[x]

df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1})
df.head()


tfidf=TfidfVectorizer(max_features=1000)

X=df['content']
Y=df['sentiment']

X_train, X_test,Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=30)

X_train=tfidf.fit_transform(X_train)
X_test=tfidf.transform(X_test)



param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}


mlflow.set_experiment("Logistic Regression Hyper Parameter tuning")

with mlflow.start_run():

    lor=LogisticRegression()
    grid_search=GridSearchCV(lor, param_grid=param_grid,scoring='f1',cv=10)
    grid_search.fit(X_train,Y_train)

    # Log the parameters
    

    # Log each parameter combination as a child run
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
            model = LogisticRegression(**params)
            model.fit(X_train, Y_train)
            
            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            
            # Print the results for verification
            print(f"Mean CV Score: {mean_score}, Std CV Score: {std_score}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

    # Log the best run details in the parent run
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)
    
    print(f"Best Params: {best_params}")
    print(f"Best F1 Score: {best_score}")

    # Save and log the notebook
    mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")




