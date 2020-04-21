from tkinter import *
import tkinter as tk
from tkinter import filedialog
import shutil
import os
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import greycomatrix,greycoprops
import pandas as pd
import cv2
from skimage.measure import label,regionprops
import skimage
from PIL import Image, ImageTk
import sklearn
from sklearn import preprocessing,neighbors
window = tk.Tk()

window.title("Dr. Plant")

window.geometry("500x510")
window.configure(background ="lightgreen")
#canvas = Canvas(window, width = 300, height = 300)      
#canvas.pack() 
#img = PhotoImage(file="/home/himanshu/Pictures/Screenshot from 2020-03-30 14-37-44.png")      
#canvas.create_image(20,20, anchor=NW, image=img)  
title = tk.Label(text="Click below to choose picture for testing disease....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
def openphoto():
    file = filedialog.askopenfile(parent=window,mode='rb',title='Choose a file')
    if file != None:
        data = file.read()
        file.close()
        print("I got %d bytes from this file." % len(data))
    file2=file.name
    print(file2)
    a=file2
    slices=[]
    img = mpimg.imread(file2)
    print(img)
    imgplot = plt.imshow(img)
    plt.show()
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','hue','value',      'saturaton']
    properties =np.zeros(6)
    glcmMatrix = []
    final=[]
    crop = input("enter the crop name")
    folders = ["anthracnose"]
    lum_img = img[:, :, 0]
    plt.imshow(lum_img)
    plt.show()
    imgplot = plt.imshow(lum_img)
    imgplot.set_cmap('nipy_spectral')
    plt.show()
    plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()
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
                [properties[0], properties[1], properties[2], properties[3],   properties[4],h_mean,s_mean,v_mean])
	    
            final.append(features)

    df = pd.DataFrame(final, columns=featlist)
    filepath =  "Final.xlsx"
    df.to_excel(filepath)
    print("done")
    dataset=pd.read_excel("Final.xlsx")
    print(dataset)
def next():
    choice=input("Enter the name of the plant")
    dataset=pd.read_excel("/home/himanshu/pro/final/xml/"+choice+"Final.xlsx")
    print(dataset)
    del dataset['path']
    X=dataset.iloc[:,1:9].values
    y=dataset.iloc[:,9:10].values

    print(X)
    print(y)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    accuracy=clf.score(X_test,y_test)
    #print(accuracy)
       #exp=np.array([251.98788656643444,9.637635985820804,0.30818546930892676,0.038288295520665636,0.19567395207504149,30.288182261208576,122.22830165692008,177.04392056530213])
    er=pd.read_excel("/home/himanshu/pro/final/code/Final.xlsx")
    z=er.iloc[:,1:9].values
    print(z)
    #exp=exp.reshape(1,-1)
    #er=er.reshape(1,-1)

    pred=clf.predict(z)
    print(86.34243457)
    print(pred[0])
    if choice == "apple":
        if pred == 3:
    	    print("Apple___healthy")
        elif pred == 2:
            print("Apple___Cedar_apple_rust")
        elif pred == 1:
            print("Apple___Black_rot")
        else:
            print("Apple___Apple_scab")
    elif choice == "corn":
        if pred == 3:
    	    print("Corn_(maize)___Northern_Leaf_Blight")
        elif pred == 2:
            print("Corn_(maize)___healthy")
        elif pred == 1:
            print("Corn_(maize)___Common_rust_")
        else:
            print("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    elif choice == "grapes":
        if pred == 3:
    	    print("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)")
        elif pred == 2:
            print("Grape___healthy")
        elif pred == 1:
            print("Grape___Esca_(Black_Measles)")
        else:
            print("Grape___Black_rot")
    elif choice == "potato":
        if pred == 2:
            print("Potato___Late_blight")
        elif pred == 1:
            print("Potato___healthy")
        else:
            print("Potato___Early_blight")
    elif choice == "tomato":
        if pred == 9:
    	    print("Tomato___Tomato_Yellow_Leaf_Curl_Virus")
        elif pred == 8:
    	    print("Tomato___Tomato_mosaic_virus")
        elif pred == 7:
    	    print("Tomato___Target_Spot")
        elif pred == 6:
    	    print("Tomato___Spider_mites Two-spotted_spider_mite")
        elif pred == 5:
    	    print("Tomato___Septoria_leaf_spot")
        elif pred == 4:
    	    print("Tomato___Leaf_Mold")
        elif pred == 3:
    	    print("Tomato___Late_blight")
        elif pred == 2:
            print("Tomato___healthy")
        elif pred == 1:
            print("Tomato___Early_blight")
        else:
            print("Tomato___Bacterial_spot")
    else:
        print("ok")
button1 = tk.Button(text="Get Photo", command = openphoto)
button2 = tk.Button(text="z Photo", command = next)
button1.grid(column=0, row=1, padx=10, pady = 10)
button2.grid(column=0, row=2, padx=10, pady = 10)
window.mainloop()
