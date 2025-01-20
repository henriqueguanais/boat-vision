from ultralytics import YOLO
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2

model = YOLO("weights/weight.pt")
classNames = ['buoy', 'cruise-ship', 'ferry-boat', 'freight-boat', 'inflatable-boat', 'kayak', 'motorboat', 'rock', 'sailboat']

def object_detect(frame):
    '''
    Faz a detecção de objetos em uma imagem.
    Parâmetros:
    -------------------
    frame: 
        Imagem a ser analisada.
    
    Retorna:
    -------------------
    frame_copy: 
        Imagem com a detecção dos objetos.
    [x1, y1, x2, y2]:
        Coordenadas do objeto detectado.
    '''
    frame_copy = np.copy(frame)
    results = model(frame_copy, conf=0.6, max_det=1)
    x1, y1, x2, y2, cls = 0, 0, 0, 0, 0
    for r in results:
        boxes = r.boxes
        if len(boxes) >= 1:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                cls = int(box.cls[0])

    org = [x1, y1]
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.putText(frame_copy, classNames[cls], org, FONT, FONTSCALE, COLOR, THICKNESS)

    return frame_copy, [x1, y1, x2, y2]


