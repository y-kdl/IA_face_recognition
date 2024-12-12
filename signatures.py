import cv2
import numpy as np
import face_recognition
import os
from os import listdir


# Global
path = 'images/'
images = [] # List of images
classNames = [] # List of image names
# Grab images from the folder
myList = listdir(path)
#print(myList)

# Load images
for cl in myList:
    #print(f'{cl}')
    curImg = cv2.imread(f'{path}{cl}')
    images.append(curImg)
    img_name = os.path.splitext(cl)[0]
    #print(img_name)
    classNames.append(img_name)

# Find face and compute encodings
def findEncodings(myImgs, names):
    encodeList = []
    value = 1
    count = 1
    for myImg, name in zip(myImgs, names):
        # Convert BGR o RGB
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_class = encode.tolist() + [name]
        encodeList.append(encode_class)
        print(f'{int((count/(len(myImgs)))*100)}% extracted ..')
        #print(encode_class)
        count += 1
    array = np.array(encodeList)
    np.save('Signatures.npy', array)
    print('Signatures saved!')
    

# generate features
encodeList = findEncodings(images, classNames)
        
    
    


    