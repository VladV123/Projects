def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

def calculeaza_colturi_patrat(centru_x, centru_y, latura):
    jumatate_latura = latura / 2
    stanga_sus_x = centru_x - jumatate_latura
    stanga_sus_y = centru_y - jumatate_latura
    dreapta_jos_x = centru_x + jumatate_latura
    dreapta_jos_y = centru_y + jumatate_latura
    return stanga_sus_x, stanga_sus_y, dreapta_jos_x, dreapta_jos_y
def compute_iou(box1, boxes):
    # Calcularea coordonatelor punctelor de intersecție
    x1 = max(box1[0], boxes[0])
    y1 = max(box1[1], boxes[1])
    x2 = min(box1[2], boxes[2])
    y2 = min(box1[3], boxes[3])

    # Calcularea suprafeței de intersecție
    intersection_area = float(max(0.0, x2 - x1)) * float(max(0.0, y2 - y1))

    # Calcularea suprafeței uniunii
    box1_area = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    boxes_area = (int(boxes[2]) - int(boxes[0])) * (int(boxes[3]) - int(boxes[1]))
    union_area = box1_area + boxes_area - intersection_area

    # Calcularea IoU
    if union_area == 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area

    return iou
def Non_Maximum_Suppression(boxes, iou_treshold=0.5):
    boxes_nms = []
    # for box in boxes
    for i in range(len(boxes)):
        is_box_valid = True  # Variabilă pentru a verifica dacă box-ul este valid
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i], boxes[j])

            if iou >= iou_treshold:
                is_box_valid = False  # Dacă IoU-ul este mare, box-ul nu este valid
                break

        if is_box_valid:
            boxes_nms.append(boxes[i])  # Adăugăm box-ul doar dacă este valid
    return boxes_nms
def algoritm_propunere_testare(data):
    propuneri_regiuni = np.zeros((566, 1500, 4), dtype=np.int16)
    nr_imagini = 0
    contor = 0
    #for images,labels in data:
    for images, _ in data:

        for image in images:

            imagine_nenormante = np.uint8(image * 255)
            keypoint_data = []
            orb = cv2.ORB_create(1500, 2)
            keypoints = orb.detect(imagine_nenormante, None)
            for keypoint in keypoints:
                x, y = keypoint.pt
                keypoint_data.append({
                    'x': x,
                    'y': y
                })
            for nr_keypoints, k in enumerate(keypoint_data):
                x = k['x']
                y = k['y']
                propuneri_regiuni[nr_imagini, nr_keypoints, :] = calculeaza_colturi_patrat(x, y, 224)
            nr_imagini = nr_imagini + 1
    lista_imagini = []
    for i, regiuni in enumerate(propuneri_regiuni):
        lista = []
        for j, regiune in enumerate(regiuni):
            x1, y1, x2, y2 = regiune
            x1_nms = max(0, x1)
            y1_nms = max(0, y1)
            x2_nms = min(x2, 3468)
            y2_nms = min(y2, 4624)
            if (x2_nms - x1_nms == 224) and (y2_nms - y1_nms == 224):
                lista.append([x1_nms, y1_nms, x2_nms, y2_nms])
    lista_imagini.append(Non_Maximum_Suppression(lista))
    return lista_imagini