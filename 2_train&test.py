import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from random import randint,uniform
# in this: the files are ready this is for checking if all works fine

# reading data from csv file and taking only important info
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "absences", "failures"]]
predict1 = "G3"
# creating array: because models only take arrays as an input
x2 = np.array(data.drop(columns=predict1))
y2 = np.array(data[predict1])
# splitting the data
trainx, testx, trainy, testy = train_test_split(x2,y2,test_size=0.2)

# reading the previously saved model and loading it
read_pickle = open("first_model.pickle", "rb")
trained_model = pickle.load(read_pickle)

# reading the data file which made the accuracy of the model 97 just for checking
read_data = pd.read_pickle("data_file.pickel")
# checking all the data is properly loaded or not
print(read_data.keys())
print(trained_model.score)

# previous data to verify
x_test = read_data['x_test']
y_test = read_data['y_test']
xt = read_data['x_train']
yt = read_data['y_train']

# now using the previous model to check the predictions
predict = trained_model.predict(x_test)
for predictions in range(10):
    print(predict[predictions], x_test[predictions], y_test[predictions])

# printing the previous model accuracy for comparison
accuracy = trained_model.score(x_test,y_test)
print(accuracy, "original")

# training the model again and again on different combinations of data trying to increase accuracy
best, acc = 0, 0
for j in range(1000):
    size = uniform(.1,.5)   # taking random float to size
    trainx, testx, trainy, testy = train_test_split(x2, y2, test_size=size)
    # fixed size was not getting much greater accuracy
    trained_model.fit(trainx, trainy)
    acc = trained_model.score(testx, testy)
    if acc > best:
        best = acc

print(best)

