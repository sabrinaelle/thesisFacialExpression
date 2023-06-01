
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
'''
emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]
flat_data_arr = []
target_arr = []
datadir = '/Users/sabrinasimkhovich/Desktop/allImages'

for i in emotions:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(emotions.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
df
print("---------------------------------------------")
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')
print("---------------------------------------------")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)
print('The Model is trained well with the given images')
model.best_params_
print("---------------------------------------------")
y_pred = model.predict(x_test)
print("The predicted Data is :")
y_pred
print("---------------------------------------------")
print("The actual data is:")
np.array(y_test)
print("---------------------------------------------")
print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
print("---------------------------------------------")
pickle.dump(model, open('img_model.p', 'wb'))
print("Pickle is dumped successfully")
print("---------------------------------------------")
model = pickle.load(open('img_model.p', 'rb'))


url = input('image URL')
img = imread(url)
img_resize = resize(img, (150, 150, 3))
l = [img_resize.flatten()]
probability = model.predict_proba(l)
for ind, val in enumerate(emotions):
    print(f'{val} = {probability[0][ind] * 100}%')
print("The predicted image is : " + emotions[model.predict(l)[0]])
print(f'Is the image a {emotions[model.predict(l)[0]]} ?(y/n)')
while (True):
    b = input()
    if (b == "y" or b == "n"):
        break
    print("please enter either y or n")

if (b == 'n'):
    print("What is the image?")
    for i in range(len(emotions)):
        print(f"Enter {i} for {emotions[i]}")
    k = int(input())
    while (k < 0 or k >= len(emotions)):
        print(f"Please enter a valid number between 0-{len(emotions) - 1}")
        k = int(input())
    print("Please wait for a while for the model to learn from this image :)")
    flat_arr = flat_data_arr.copy()
    tar_arr = target_arr.copy()
    tar_arr.append(k)
    flat_arr.extend(l)
    tar_arr = np.array(tar_arr)
    flat_df = np.array(flat_arr)
    df1 = pd.DataFrame(flat_df)
    df1['Target'] = tar_arr
    model1 = GridSearchCV(svc, param_grid)
    x1 = df1.iloc[:, :-1]
    y1 = df1.iloc[:, -1]
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.20, random_state=77, stratify=y1)
    d = {}
    for i in model.best_params_:
        d[i] = [model.best_params_[i]]
    model1 = GridSearchCV(svc, d)
    model1.fit(x_train1, y_train1)
    y_pred1 = model.predict(x_test1)
    print(f"The model is now {accuracy_score(y_pred1, y_test1) * 100}% accurate")
    pickle.dump(model1, open('img_model.p', 'wb'))
    print("Thank you for your feedback")'''
