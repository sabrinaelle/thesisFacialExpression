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


datadir = r"C:\Users\Sab\PycharmProjects\dlibfixmaybe\Training_Images"
emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]
# flat_data_arr = []
# target_arr = []
detector = dlib.get_frontal_face_detector()
flat_file = open("flat_data_arr_fl.p", 'rb')
flat_data_arr = pickle.load(flat_file)
tar_file = open("target_arr_fl.p", 'rb')
target_arr = pickle.load(tar_file)



# for i in emotions:
#     print(f'loading... category : {i}')
#     path = os.path.join(datadir, i)
#     for img in os.listdir(path):
#         if not img.startswith('.') and os.path.isfile(os.path.join(path, img)):
#             img_array = imread(os.path.join(path, img))
#             while (True):
#                 faces = detector(img_array)
#                 for face in faces:
#                     x1 = face.left()
#                     y1 = face.top()
#                     x2 = face.right()
#                     y2 = face.bottom()
#                 x, y = x1, y1
#                 width, height = x2, y2
#                 area = (x, y, width, height)
#                 image = Image.fromarray(img_array, 'RGB')
#                 break
#             cropped_img = image.crop(area)
#             imgARR = np.array(cropped_img)
#             grayimg = np.mean(imgARR, axis=2)
#             img_resized = resize(grayimg, (150, 150, 3))
#             flat_data_arr.append(img_resized.flatten())
#             target_arr.append(emotions.index(i))
#     print(f'loaded category:{i} successfully')
#
# with open("flat_data_arr_fl.p", 'wb') as fi:
#     pickle.dump(flat_data_arr, fi)
# with open("target_arr_fl.p", 'wb') as fi:
#     pickle.dump(target_arr, fi)


print("Graphing data")
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
# print("The training of the model is started, please wait for while as it may take few minutes to complete")
# print("---------------------------------------------")
# model = svm.SVC(kernel="linear", probability=True)
# print("model created")
# print("data is now being fit to model...")
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# np.array(y_test)
# print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
# print("---------------------------------------------")
# pickle.dump(model, open('img_model.p', 'wb'))

while (True):
    model = pickle.load(open("img_model.p", "rb"))
    print("model loaded")
    y_pred = model.predict(x_test)
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
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
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
    print("The predicted image is : " + emotions[int(model.predict(l)[0])])
    print(f'Is the image a {emotions[int(model.predict(l)[0])]} ?(y/n)')
    while (True):
        b = input()
        if (b == "y" or b == "n"):
            break
        print("please enter either y or n")
    # if answer is "n" or NO, the image is re trained with the new data
    if (b == 'y'):
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
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.30, random_state=85, stratify=y1)
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
        model1 = svm.SVC(kernel="linear", probability=True)
        x1 = df1.iloc[:, :-1]
        y1 = df1.iloc[:, -1]
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.20, random_state=77, stratify=y1)
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
