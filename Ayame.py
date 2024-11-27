from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def load_data(n,m):
  iris=load_iris()
  X=np.insert(iris.data[n:m,:],0,1,axis=1)#numpy.insert(arr, obj, values, axis=None)
  Y=iris.target[n:m]
  Y=Y*2-3
  return X, Y

def projection(W, X):
  z=W.dot(np.transpose(X))
  return z

def Loss_Function(e):
  #Calculate squared error using inner product
  L=e.dot(np.transpose(e))
  return L

def classify(z):
  C = np.where(z<0,-1,1)
  return C


X, y = load_data(50,150)
a = 0.000001
temp=0
N=100
I=0

#Define the parameters
W=np.zeros(5)
W[1:5]=1

#Project the features to the plain
z=projection(W, X)

#Gradient Descendant
#Calculate errors
e=z-y
L_prev=Loss_Function(e)
History=np.zeros(N)
History[0]=L_prev
for i in range(0,N):
  I+=1
  for j in range(0,5):
    W[j]=W[j]-2*a*projection(e,X[:,j])
  z=projection(W,X)
  e=z-y
  L = Loss_Function(e)
  History[i]=L
  variation=L-L_prev
  if abs(variation)<=0.0001:
    temp=1
    break

  L_prev=L


if temp==1:
  print("収束しました！",L)
else:
  print("収束せず",L)

#https://qiita.com/s_wing/items/f7d9db1a6a753aca76ae
C=classify(z)
count=np.sum(C==y)
print(X.shape[0])
print("正解数:",count,"正解率:",count/X.shape[0])
print("終了パラメータ",W)

index=int(random.random() * (100.0+1))
name =  "Versicolor" if y[index]==1 else "Virginica"
print("X:",X[index,:],"name:",name)
prediction=classify(projection(W,X[index]))
print("number",index,"was classified into",prediction, "Versicolor" if prediction==1 else "Virginica")


#損失関数の描画
t=np.arange(0,I)
plt.figure(figsize=(8,6))
plt.plot(t,History[0:I],label="Loss Function",linestyle="-", linewidth=2)
plt.xlabel("x")
plt.ylabel("L")
plt.legend()
plt.grid(True)
plt.show()

