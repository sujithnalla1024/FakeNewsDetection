from django.shortcuts import render
import pickle
from django.shortcuts import render
import pandas as pd
import urllib
from newspaper import Article
data = pd.read_csv('C:/Users/91965/Downloads/news.csv')
modell = pickle.load(open('models/modelObject.pkl','rb'))
with open('models/modelObject.pkl','rb') as handle:
    model = pickle.load(handle)
def home(request):
  return render(request,'home.html');
def fromsubmit(request):
  url=[request.GET['urltext']][0]
  url = urllib.parse.unquote(url)
  article = Article(str(url))
  article.download()
  article.parse()
  article.nlp()
  news = article.summary
  pred = modell.predict([news])
  prediction = pred[0]
  context = {'result':prediction}
  #if(prediction == 'REAL' )
  if (context['result'] == 'REAL'):
    return render(request,'real.html')
  else:
    return render(request,'fake.html')