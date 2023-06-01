from PIL import Image
import dlib
import pandas as pd
from sklearn import svm
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from skimage.transform import resize
from SVMmodelcreation import train_test

class emotionDetector:
    model = pickle.load(open("img_model.p", "rb"))
    flat_data_arr = pickle.load(open("flat_data_arr_fl.p", 'rb'))
    target_arr = pickle.load(open("target_arr_fl.p", 'rb'))
    emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]
    detector = dlib.get_frontal_face_detector()

    ask = input("Would you like to test an image? y/n")
    if ask == "y":
        print("Great! How many images would you like to test?")
        num = input("enter a number: ")
        for i in num:
            print("model loaded")
            y_pred = model.predict(train_test()[1])
            print(f"The model is {accuracy_score(y_pred, train_test()[3]) * 100}% accurate")
            url = input('Enter URL of Image')
            img = imread(url)
            while (True):
                faces = detector(img)
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                x, y = x1, y1
                width, height = x2, y2
                area = (x, y, width, height)
                image = Image.fromarray(img, 'RGB')
                break
            cropped_img = image.crop(area)
            croppedImgArr = np.array(cropped_img)
            grayomginput = np.mean(croppedImgArr, axis=2)
            inputRES = resize(grayomginput, (
                150, 150, 3))  # probably messing up the resize becuase addimg more area. but probably not gonna fix it
            l = [inputRES.flatten()]
            probability = model.predict_proba(l)
            print(f"The model is {accuracy_score(y_pred, train_test()[3]) * 100}% accurate")
            print("The predicted image is : " + emotions[int(model.predict(l)[0])])
            print(f'Is the image a {emotions[int(model.predict(l)[0])]} ?(y/n)')
            while (True):
                b = input()
                if (b == "y" or b == "n"):
                    break
                print("please enter either y or n")
            if b == 'y':
                print("Nice! The data will be added to the model and updated")
                flat_arr = flat_data_arr.copy()
                tar_arr = target_arr.copy()
                tar_arr = np.append(tar_arr, int(model.predict(l)[0]))
                flat_arr.extend(l)
                tar_arr = np.array(tar_arr)
                flat_df = np.array(flat_arr)
                df1 = pd.DataFrame(flat_df)
                df1['Target'] = tar_arr
                model1 = svm.SVC(kernel="linear", probability=True)
                x1 = df1.iloc[:, :-1]
                y1 = df1.iloc[:, -1]
                x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.30, random_state=85,
                                                                        stratify=y1)
                model1.fit(x_train1, y_train1)
                print("The previous model is now updated with your previous image!")
                y_pred1 = model1.predict(x_test1)
                print(f"The model is now {accuracy_score(y_pred1, y_test1) * 100}% accurate")
                with open('img_model.p', "wb") as model_file:
                    pickle.dump(model1, model_file)
                with open("flat_data_arr_fl.p", "ab") as flat_file:
                    pickle.dump(flat_arr, flat_file)
                with open("target_arr_fl.p", "ab") as tar_file:
                    pickle.dump(tar_arr, tar_file)
                print("all files updated")

            if b == 'n':
                print("What is the image?")
                for i in range(len(emotions)):
                    print(f"Enter {i} for {emotions[i]}")
                k = int(input())
                while k < 0 or k >= len(emotions):
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
                model1 = svm.SVC(kernel="linear", probability=True)
                x1 = df1.iloc[:, :-1]
                y1 = df1.iloc[:, -1]
                x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.20, random_state=77,
                                                                        stratify=y1)
                model1.fit(x_train1, y_train1)
                y_pred1 = model1.predict(x_test1)
                print(f"The model is now {accuracy_score(y_pred1, y_test1) * 100}% accurate")
                with open('img_model.p', "wb") as model_file:
                    pickle.dump(model1, model_file)
                with open("flat_data_arr_fl.p", "ab") as flat_file:
                    pickle.dump(flat_arr, flat_file)
                with open("target_arr_fl.p", "ab") as tar_file:
                    pickle.dump(tar_arr, tar_file)
                print("all files updated")




