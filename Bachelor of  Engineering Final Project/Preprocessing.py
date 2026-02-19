import os
import cv2
import json
import numpy as np
import albumentations as alb

augmentor = alb.Compose([], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))
for partition in ['antrenare', 'testare', 'validation']:
    for image, labels in zip(os.listdir(os.path.join(r"...\Clasa_other\Poze_pasare_dupa cropare",'poze')),os.listdir(os.path.join(r"...\Clasa_other\Poze_pasare_dupa cropare", 'eticheta'))):
        img = cv2.imread(os.path.join(r"...\Clasa_other\Poze_pasare_dupa cropare",'poze', image))
        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join(r"...\Clasa_other\Poze_pasare_dupa cropare",'eticheta',labels)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
    #
        coords[0] = label['shapes'][0]['points'][0][0]
        coords[1] = label['shapes'][0]['points'][0][1]
        coords[2] = label['shapes'][0]['points'][1][0]
        coords[3] = label['shapes'][0]['points'][1][1]
        coords = list(np.divide(coords, [3468, 4624, 3468, 4624]))

        augmented = augmentor(image=img, bboxes=[coords], class_labels=[label['shapes'][0]['label']])

        annotation = {}
        annotation['image'] = image

        if os.path.exists(label_path):
            if len(augmented['bboxes']) == 0:
                annotation['bbox'] = [0, 0, 0, 0]
                annotation['class'] = 4
            elif (label['shapes'][0]['label'] == "diji_mini2"):
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 0
            elif (label['shapes'][0]['label'] == "mavic_air"):
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 1
            elif (label['shapes'][0]['label'] == "Phantom"):
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 2
            elif (label['shapes'][0]['label'] == "tello"):
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 3
            elif (label['shapes'][0]['label'] == "other"):
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 5
        else:
            annotation['bbox'] = [0, 0, 0, 0]
            annotation['class'] = 4

        with open(os.path.join(r"...\Clasa_other\Poze_pasare_dupa cropare\eitcheta_dupa_anotation", f'{image.split(".")[0]}.json'), 'w') as f:
            json.dump(annotation, f)