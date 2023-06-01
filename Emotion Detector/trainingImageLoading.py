from PIL import Image
import dlib
from skimage.io import imread
import numpy as np
import os
import pickle
from skimage.transform import resize


class trainingImageLoading:
    datadir = r"C:\Users\Sab\PycharmProjects\dlibfixmaybe\Training_Images"
    emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]

    flat_data_arr = []
    target_arr = []
    detector = dlib.get_frontal_face_detector()

    for i in emotions:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            if not img.startswith('.') and os.path.isfile(os.path.join(path, img)):
                img_array = imread(os.path.join(path, img))
                while (True):
                    faces = detector(img_array)
                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()
                    x, y = x1, y1
                    width, height = x2, y2
                    area = (x, y, width, height)
                    image = Image.fromarray(img_array, 'RGB')
                    break
                cropped_img = image.crop(area)
                imgARR = np.array(cropped_img)
                grayimg = np.mean(imgARR, axis=2)
                img_resized = resize(grayimg, (150, 150, 3))
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(emotions.index(i))
        print(f'loaded category:{i} successfully')

    with open("flat_data_arr_fl.p", 'wb') as fi:
        pickle.dump(flat_data_arr, fi)
    with open("target_arr_fl.p", 'wb') as fi:
        pickle.dump(target_arr, fi)
