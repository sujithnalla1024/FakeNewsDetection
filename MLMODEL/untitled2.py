import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle

dataset = pd.read_csv('news.csv')
x = dataset['text']
y = dataset['label']

dataset.head()

dataset.isnull().any()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train  = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

pipeline = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('nbmodel',MultinomialNB())])

pipeline.fit(x_train,y_train)

score = pipeline.score(x_test,y_test)
print('accuracy',score)

pred = pipeline.predict(x_test)

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))

with open('model.pkl','wb') as handle:
  pickle.dump(pipeline,handle,protocol=pickle.HIGHEST_PROTOCOL)