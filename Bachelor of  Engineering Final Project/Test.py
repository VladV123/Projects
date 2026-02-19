import tensorflow as tf
import json
import numpy as np
import cv2
import os
import Classification_model
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
import utils
import time


t1 = time.time()
imagini_testare = tf.data.Dataset.list_files(r"C:\Users\Aorus\Desktop\Test\imagini\*.jpeg", shuffle=False)
imagini_testare = imagini_testare.map(load_image)
imagini_testare = imagini_testare.map(lambda x: x / 255)
etichete_testare = tf.data.Dataset.list_files(r"C:/Users/Aorus/Desktop/Test/label/*.json", shuffle=False)
etichete_testare = etichete_testare.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))
testare = tf.data.Dataset.zip((imagini_testare, etichete_testare))
testare = testare.batch(2)
model = load_model(r'C:\Users\Aorus\Desktop\Licenta\pythonProject_CNN\model_versiunea_finala_avion_fara_pasare_dr_0.6_initial_l_r_0.001_early_stopping_patience_3_adam.h5')
data = testare.as_numpy_iterator()
coordonate = np.zeros((283,2,4),dtype=np.uint16)
clss = 4 * np.ones((283,2,1),dtype=np.uint8)
lista_regiuni = utils.algoritm_propunere_testare(data)
contor = 0
data = testare.as_numpy_iterator()
for i,(images,labels) in enumerate(data):
    print("batch nr:",i)
    for k,imagine in enumerate(images):
        print("elementul nr:",k)
        for j,sectiune in enumerate(lista_regiuni[contor]):
            print("nr_sectiune,nr imagine:",j,contor)
            x1,y1,x2,y2 = sectiune
            image_to_predict = np.expand_dims(imagine[y1:y2, x1:x2], axis=0)
            clasa = model.predict(image_to_predict)
            if np.argmax(clasa) != 4:
                print(clasa)
                clss[i,k,:] = np.argmax(clasa)
                coordonate[i,k,:] = [x1 ,y1 ,x2 ,y2]
            contor = contor + 1
            print(contor)
clasa_prezisa =[]
clasa_true = []
t2 = time.time()
durata_executiei = t2-t1
print(durata_executiei)
date = testare.as_numpy_iterator()
for i,(imagine,labels) in enumerate(date):
    print("batch:",i)
    for j, (clasa, puncte) in enumerate(zip(labels[0], labels[1])):
        clasa_prezisa.append(clss[i,j,:])
        clasa_true.append(clasa)
cm = confusion_matrix(clasa_true, clasa_prezisa)
recall = recall_score(clasa_true, clasa_prezisa, average='macro')
precision = precision_score(clasa_true, clasa_prezisa, average='macro')
f1 = f1_score(clasa_true, clasa_prezisa, average='macro')
print('Recall:', recall)
print('Precision:', precision)
print('F1 score:', f1)
print(cm)