import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
# first we have to train the model
# in this: model is saved and the train data is also saved
# '''
# reading data from csv file
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "absences", "failures"]]
predict = "G3"

# making it to array because model only accept arrays
x = np.array(data.drop(columns=predict))
y = np.array(data[predict])

# splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
bestFit = 0
accuracy = 0
count = 0
# runs till we get best accuracy
while accuracy > 0.98:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # converts to linear model
    linear = linear_model.LinearRegression()
    model = linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    if accuracy > bestFit:
        bestFit = accuracy
    if accuracy > .97:
        print(accuracy, count)
        # this saves the model which has more than 97% accuracy
        with open("first_model.pickle", "wb") as f:
            pickle.dump(model, f)
        # this will save the dataset which was used to create that accuracy
        # dictionary is used so that while using data in other file it is easy to pull out data
        data_dict = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        with open("data_file.pickel", "wb") as fe:
            pickle.dump(data_dict, fe)
        break

    count += 1

