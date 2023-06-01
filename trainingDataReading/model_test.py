from skimage.transform import resize
from skimage.io import imread
import pickle
import os

emotions = ["Neutral", "Happy", "Sad", "Surprise", "Anger", "Disgust", "Fear"]



model = pickle.load(open("img_model.p","rb"))
#url=input('Enter URL of Image')
#img=imread(url)
file = input("please enter a file of test images: ")
for img in os.listdir(file):
    img_array = imread(os.path.join(file, img))
    img_resize=resize(img_array,(150,150,3))
    l=[img_resize.flatten()]
    probability=model.predict_proba(l)
    #for ind,val in enumerate(emotions):
     #   print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : "+emotions[model.predict(l)[0]])
    print(img + "-->")
    #print(f'The predicted image is {emotions[model.predict(l)[0]]}')

