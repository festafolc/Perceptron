#Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 

data = pd.read_csv("spam.csv", encoding='latin-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
data = data.rename(columns={'v1': 'type','v2': 'sms'})
x= data.sms
y=data.type
vectorizer = TfidfVectorizer(stop_words={'english'}, strip_accents='unicode')
x1 = vectorizer.fit_transform(x)
feature_names = vectorizer.get_feature_names()
x= x1.toarray()

# Divisão do dataset 70/30 (treino/teste)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 

thetaZero=0 
epoch=150
e = 0
learnRate=0.001
total_error=0
errors=[]
accuracy=[]
thetaVector=np.zeros(len(feature_names))

#Perceptrão - Treino
for t in range(epoch):
    total_error=0
    for i in range(len(y_train)):
        if y_train.iloc[i]=="ham":
          labelY= 1
        if y_train.iloc[i]=="spam":
          labelY= -1
        if (labelY * (np.dot(thetaVector, x_train[i]) + thetaZero) <= 0) :
            thetaVector = thetaVector + learnRate*x_train[i] * labelY 
            thetaZero = thetaZero + labelY
            total_error = total_error+1
    errors.append(total_error)
    e = e + 1
    print("Epoch:", e)
    acc = (len(y_train) - total_error) / len(y_train)
    accuracy.append(acc)
    print(acc)

# Perceptrão - Teste
num_error=0
prod_vector=0
for i in range(len(y_test)):
  
  if y_train.iloc[i]=="ham":
    labelY = 1
  if y_train.iloc[i]=="spam":
    labelY =-1

  if (labelY * (np.dot(thetaVector, x_test[i]) + thetaZero) <= 0) :  
    num_error = num_error+1

#Gráficos 
plt.title("Evolução do número de erros")
plt.plot(errors)
plt.xlabel("Número de iteracoes")
plt.ylabel("Número de erros")
plt.show()
plt.title("Evolução da precisão")
plt.plot(accuracy)
plt.xlabel("Número de iteracoes")
plt.ylabel("Precisão")
plt.show()
print()
print("Precisão do algoritmo Perceptrão na amostra reservada para os teste:", (len(y_test)-num_error)/len(y_test))
