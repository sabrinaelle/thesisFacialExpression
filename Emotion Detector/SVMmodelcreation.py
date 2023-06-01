import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def train_test(self):
    df = pd.DataFrame(SVMmodelcreation.flat_data_arr)
    df['Target'] = SVMmodelcreation.target_arr
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
    return x_train, x_test, y_train, y_test

class SVMmodelcreation:
    flat_file = open("flat_data_arr_fl.p", 'rb')
    flat_data_arr = pickle.load(flat_file)
    tar_file = open("target_arr_fl.p", 'rb')
    target_arr = pickle.load(tar_file)

    model = svm.SVC(kernel="linear", probability=True)
    model.fit(train_test()[0], train_test()[2])
    y_pred = model.predict(train_test()[1])
    np.array(train_test()[3])
    # print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
    pickle.dump(model, open('img_model.p', 'wb'))

