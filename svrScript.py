######import required libraries
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split #split data into train/test
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv") #load data values
#create train and test data
data = df.values
X = data[:, 1:]  # all rows, no label
y = data[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Create SVR model 
model = SVR(C=1000, cache_size=500, epsilon=0.5, kernel='rbf')
#Fit the previously created model
print("Fitting SVR model this could take some time...")
model.fit(X_train, y_train)
#Add test values to the plot
trainlen = len(y_train)
plt.plot([i+trainlen for i in range(len(X_test[:,1]))], y_test)
#ADD predicted values for the test partition to the plot
plt.plot([i+trainlen for i in range(len(X_test[:,1]))], model.predict(X_test))
#show the plot
plt.show()


"""
####use the code below to perform different calculations for different values of epsiolon and C parameters
##try different epsilon values
epsilons = np.arange(1, 9)
scores = []
for e in epsilons:
	print("Checking epsilon: " + str(e))
	model.set_params(epsilon=e)
	model.fit(X_train, y_train)
	score=model.score(X_test, y_test)
	print("Score is: " +str(score))
	scores.append(score)
plt.plot(epsilons, scores)
plt.title("Epsilon effect")
plt.xlabel("epsilon")
plt.ylabel("score")
plt.show()

##try different Cost values
model.set_params(epsilon=5)
Cs = [1e0, 1e1, 1e2, 1e3]
scores = []
for c in Cs:
    model.set_params(C=c)
    model.fit(X_train, y_train)
    score=model.score(X_test, y_test)
    print("Score is: " +str(score))
    scores.append(score)
plt.plot(Cs, scores)
plt.title("C effect")
plt.xlabel("C")
plt.ylabel("score")
plt.show()
"""
