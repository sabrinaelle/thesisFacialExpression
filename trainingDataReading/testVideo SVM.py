# Sabrina Simkhovich sabrinasimkhovich@gmail.com

# import necessary libraries
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# define necessary classifications --> correspond with training image files and directory
datadir = "/Users/sabrinasimkhovich/Desktop/all_Emotion_Images"
emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]
flat_data_arr=[]
target_arr=[]

# SVM variables
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)

# pulls files from image directory and creates data and target arrays that will be used for testing, training, and fitting

for i in emotions:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        if not img.startswith('.') and os.path.isfile(os.path.join(path, img)):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(emotions.index(i))
    print(f'loaded category:{i} successfully')


# open previous saved data files so to not need to load the images every time the code is ran. Saves about five mintues
#flat_file = open("flat_data_arr_fl", 'rb')
#flat_data_arr = pickle.load(flat_file)
#tar_file = open("target_arr_fl", 'rb')
#target_arr = pickle.load(tar_file)

# saving original data to seperate files
'''
with open("flat_data_arr_fl", 'wb') as fi:
    pickle.dump(flat_data_arr, fi)
with open("target_arr_fl", 'wb') as fi:
    pickle.dump(target_arr, fi)
'''

# creates an arrat for both input and target data
#flat_data = np.array(flat_data_arr)
#target = np.array(target_arr)

# creates a plotted graph of data
df = pd.DataFrame(flat_data_arr)
df['Target'] = target_arr

print("---------------------------------------------")
# setting x and y variables to a certain data point --> pretty sure its the end
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# randomly splits data frame into train and test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# trains the model and creates a prediction variable
'''
print("The training of the model is started, please wait for while as it may take few minutes to complete")
print("---------------------------------------------")
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)
model.best_params_
y_pred = model.predict(x_test)
np.array(y_test)
print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
print("---------------------------------------------")
pickle.dump(model, open('img_model.p', 'wb'))
'''

# currently set up as a loop for testing
# loads current model, gets URL input from user, and prints out predicted emotion along with accuracy score.
ans = input("Would you like to test an image? (y/n)")
while ans == "y":
    model = pickle.load(open("img_model.p", "rb"))
    y_pred = model.predict(x_test)
    url = input('Enter URL of Image')
    img = imread(url)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)
    # for ind, val in enumerate(emotions):
    #    print(f'{val} = {probability[0][ind] * 100}%')
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
    print("The predicted image is : " + emotions[model.predict(l)[0]])
    print(f'Is the image a {emotions[model.predict(l)[0]]} ?(y/n)')
    pickle.dump(model, open('img_model.p', 'wb'))
    while (True):
        b = input()
        if (b == "y" or b == "n"):
            break
        print("please enter either y or n")
# if answer is "n" or NO, the image is re trained with the new data
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
        tar_arr = np.append(tar_arr, k)
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
        # all updated variables are saved to files for safety
        with open ('img_model.p', "wb") as model_file:
            pickle.dump(model1, model_file)
        with open("flat_data_arr_fl", "wb") as flat_file:
            pickle.dump(flat_arr, flat_file)
        with open("target_arr_fl", "wb") as tar_file:
            pickle.dump(tar_arr, tar_file)
    #pickle.dump(model, open('img_model.p', 'wb'))
