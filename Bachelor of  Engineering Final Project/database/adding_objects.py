import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
    
def rotire_cu_padding(imagine, unghi_rotatie):

    inaltime, latime = imagine.shape[:2]
    
    centru = (latime // 2, inaltime // 2)
    matrice_rotatie = cv2.getRotationMatrix2D(centru, unghi_rotatie, 1)
    
    cos = np.abs(matrice_rotatie[0, 0])
    sin = np.abs(matrice_rotatie[0, 1])
    
    new_width = int((latime * cos) + (inaltime * sin))
    new_height = int((latime * sin) + (inaltime * cos))
    
    matrice_rotatie[0, 2] += (new_width / 2) - centru[0]
    matrice_rotatie[1, 2] += (new_height / 2) - centru[1]
    
    img_rotita_cu_padding = cv2.warpAffine(imagine, matrice_rotatie, (new_width, new_height),
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return img_rotita_cu_padding 

director = r'C:\Users\Aorus\Desktop\Clasa_other\Background' 
list_dir = os.listdir(director) 
for i in range(50): 
    src1 = cv2.imread(r'C:\Users\Aorus\Desktop\Clasa_other\Poze_pasare\pasarea13.png', cv2.IMREAD_UNCHANGED)
    dst = cv2.imread(os.path.join(r"C:\Users\Aorus\Desktop\Clasa_other\Background", random.choice(list_dir)))
    x = random.randint(80,120) 
    y = int(x/1.2) 
    src1 = cv2.resize(src1, (x, y), interpolation=cv2.INTER_NEAREST) 
    h_flip = random.randint(0, 1) 
    unghi_rotatie = random.randint(-15,-15) 
    src1 = rotire_cu_padding(src1,unghi_rotatie) 
    src1 = horizontal_flip(src1, h_flip) 
    src_mask = src1[:, :, 3] 
    if src1 is not None: 
        canal_rosu = src1[:, :, 2]  # Canalul Roșu 
        canal_verde = src1[:, :, 1]  # Canalul Verde 
        canal_albastru = src1[:, :, 0]  # Canalul Albastru 
        src = cv2.merge([canal_albastru, canal_verde, canal_rosu]) 
    src_mask =cv2.merge([src_mask, src_mask, src_mask]) 
    src_mask = np.uint8(src_mask/255) 
    invert = cv2.bitwise_not(src_mask) 
    invert = np.uint8(invert/255) 
    imgFin = np.multiply(src_mask, src) 
    x_pos = random.randint(900, 3000) 
    y_pos = random.randint(300, 1500) 
    dst[y_pos:y_pos + invert.shape[0], x_pos:x_pos + invert.shape[1]] = np.multiply(dst[y_pos:y_pos + invert.shape[0], x_pos:x_pos + invert.shape[1]], invert)
    result = cv2.addWeighted(dst[y_pos:y_pos + imgFin.shape[0], x_pos:x_pos + imgFin.shape[1]], 1, imgFin, 1, 0)
    dst[y_pos:y_pos + imgFin.shape[0], x_pos:x_pos + imgFin.shape[1]] = result 
    director_save = rf"C:\Users\Aorus\Desktop\Clasa_other\Poze_pasare_dupa cropare\testare\pasarea13_{i}.jpeg"
    cv2.imwrite(director_save,dst)