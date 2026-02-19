import os
import cv2
from matplotlib import pyplot as plt
import json
import numpy as np
from numpy import random
import utils

director_de_baza = r"...\Sliding_window\Train_full"
for images, labels in zip(os.listdir(os.path.join(director_de_baza, "poze")), os.listdir(os.path.join(director_de_baza, "etichete_poze"))):
    img = cv2.imread(os.path.join(os.path.join(director_de_baza, "poze"), images))
    print(images)
    labels = utils.load_labels(os.path.join(os.path.join(director_de_baza, "etichete_poze"), labels))
    nume_imagini.append(images)
    imagini.append(img)
    nr_regiuni = 0
    d = 0
    best_score = 0
    for a in range(0, 3468 - 224, 10):
        for b in range(0, 4624 - 224, 10):
            if utils.compute_iou(labels[1], [a / 3468, b / 4624, (a + i) / 3468, (b + i) / 4624]) >= 0.15:
                if utils.compute_iou(labels[1], [a / 3468, b / 4624, (a + i) / 3468, (b + i) / 4624]) > best_score:
                    best_score = utils.compute_iou(labels[1], [a / 3468, b / 4624, (a + i) / 3468, (b + i) / 4624])
                    best_coordonate = [a, b, (a + i), (b + i)]
                    best_clasa = labels[0]
    score[numar_poza, nr_regiuni, :] = best_score
    coordonate_regiuni[numar_poza, nr_regiuni, :] = best_coordonate
    clasa_regiuni[numar_poza, nr_regiuni, :] = best_clasa
    nr_regiuni = nr_regiuni + 1
    while nr_regiuni < nr_max_regiuni:
        a = random.randint(3468)
        b = random.randint(4624)

        if utils.compute_iou(labels[1], [a / 3468, b / 4624, (a + i) / 3468, (b + i) / 4624]) < 0.07 and a < 3468 - 224 and b < 4624 - 224:
            score[numar_poza, nr_regiuni, :] = utils.compute_iou(labels[1], [a / 3468, b / 4624, (a + i) / 3468, (b + i) / 4624])
            coordonate_regiuni[numar_poza, nr_regiuni, :] = [a, b, (a + i), (b + i)]
            clasa_regiuni[numar_poza, nr_regiuni, :] = [4]
            nr_regiuni = nr_regiuni + 1
            numar_poza = numar_poza + 1

for nr, poza in enumerate(imagini):
    clase, boxes = utils.Non_Maximum_Suppression(coordonate_regiuni[nr, :, :], clasa_regiuni[nr, :, :], score[nr, :, :])
    nr_reg = 0
    print(clase)
    for box, clasa in zip(boxes, clase):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        
        cv2.imwrite(os.path.join(director_de_baza, "sectiuni_pasari", f"sectiunea{nr_reg}_" + f'{nume_imagini[nr].split(".")[0]}.jpg'), poza[y1:y2, x1:x2])

        x1 = box[0] / 3468
        y1 = box[1] / 4624
        x2 = box[2] / 3468
        y2 = box[3] / 4624

        data = {
            "bbox": [x1, y1, x2, y2],
            "class": clasa[0]
        }
        with open(os.path.join(director_de_baza, "etichete_sectiuni_pasari", f"sectiunea{nr_reg}_" +  f'{nume_imagini[nr].split(".")[0]}.json'),"w") as json_file:
            json.dump(data, json_file, indent=4)
        nr_reg = nr_reg + 1 