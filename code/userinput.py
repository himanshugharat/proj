import numpy as np
import os
import matplotlib.pyplot as pyplot
from skimage.feature import greycomatrix,greycoprops
import pandas as pd
import cv2
from skimage.measure import label,regionprops
import skimage
slices=[]
proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','hue','value', 'saturaton']
properties =np.zeros(6)
glcmMatrix = []
final=[]
crop = input("enter the crop name")
a=input("enter the path of the image")
folders = ["anthracnose"]
# folders = ["blight","rust","mildew"]
for folder in folders:
    #print()
    #labell=folders.index(folder)
    #INPUT_SCAN_FOLDER="/home/himanshu/pro/Leaf-Disease-Detection/data/rose/"+folder+"/"

    #image_folder_list = os.listdir("/home/himanshu/pro/Leaf-Disease-Detection/data/rose/"+folder+"/")

    for i in range(1):

        abc =cv2.imread(a)

        gray_image = cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(abc, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        h_mean = np.mean(h)
        h_mean = np.mean(h_mean)

        s_mean = np.mean(s)
        s_mean = np.mean(s_mean)

        v_mean = np.mean(v)
        v_mean = np.mean(v_mean)



        # images = images.f.arr_0
        #print(image_folder_list[i])


        glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
        for j in range(0, len(proList)):
            properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

        features = np.array(
            [properties[0], properties[1], properties[2], properties[3], properties[4],h_mean,s_mean,v_mean])
        final.append(features)

df = pd.DataFrame(final, columns=featlist)
filepath =  "Final.xlsx"
df.to_excel(filepath)
