import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


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

vectorizer= {
    'BOW':CountVectorizer(max_features=1000),
    'TFIDF': TfidfVectorizer(max_features=1000)
}

models={
    'Logistic Regression':LogisticRegression(),
    'Gradient Boosting':GradientBoostingClassifier(),
    'XgBoost':XGBClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost':AdaBoostClassifier()   

}

import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/nmathew23/mlops-mini-project.mlflow")
dagshub.init(repo_owner='nmathew23', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_experiment("BoW vs TFidf")

X=df['content']
Y=df['sentiment']

X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=20)


# Start the parent run

with mlflow.start_run(run_name="All experiments") as parent_run:
    # Loop through each algorithm and feature extraction method as child runs 
    for model_name,model in models.items():
        for vec_name, vec in vectorizer.items():
            with mlflow.start_run(run_name=f'{model_name} with {vec_name}',nested=True) as child_run:
                X_train_tran=vec.fit_transform(X_train)
                X_test_tran=vec.transform(X_test)

                 # Log preprocessing parameters
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", model_name)
                mlflow.log_param("test_size", 0.2)

                
                model.fit(X_train_tran,Y_train)

                # Log the model parameters
                # Log model parameters
                if model_name == 'LogisticRegression':
                    mlflow.log_param("C", model.C)                
                elif model_name == 'XGBoost':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                elif model_name == 'RandomForest':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                    mlflow.log_param("max_features", model.max_features)
                elif model_name == 'GradientBoosting':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                elif model_name == 'AdaBoost':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
            
                # Model Evaluation

                y_pred=model.predict(X_test_tran)
                accuracy = accuracy_score(Y_test, y_pred)
                precision = precision_score(Y_test, y_pred)
                recall = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                # Log evaluation Metric

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)


                #Log the dataset
                #mlflow.log_input(df.to_csv('Data.csv',index=False))

                # Log the file
                mlflow.log_artifact(__file__)

                # # Print the results for verification
                print(f"Algorithm: {model_name}, Feature Engineering: {vec_name}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")

