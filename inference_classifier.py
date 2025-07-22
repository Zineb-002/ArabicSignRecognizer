# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 17:21:06 2025

@author: zineb
"""

import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image  # <-- Import Pillow

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'ا', 11: 'ب', 12: 'ت', 13: 'ث', 14: 'ج', 15: 'ح', 16: 'خ', 17: 'د', 18: 'ذ',
    19: 'ر', 20: 'ز', 21: 'س', 22: 'ش', 23: 'ص', 24: 'ض', 25: 'ط', 26: 'ظ', 27: 'ع',
    28: 'غ', 29: 'ف', 30: 'ق', 31: 'ك', 32: 'ل', 33: 'م', 34: 'ن', 35: 'ه', 36: 'و', 37: 'ي'
}

# Charge une police compatible arabe — adapte le chemin vers le fichier .ttf sur ta machine
font_path = "arial.ttf"  # <-- Remplace par le chemin complet d'une vraie police arabe si besoin
font_size = 40
font = ImageFont.truetype(font_path, font_size)

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # Dessine le rectangle avec OpenCV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Convertir frame (OpenCV BGR) en PIL Image RGB
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Position du texte (au-dessus du rectangle)
        text_position = (x1, max(y1 - 50, 0))  # pour éviter coordonnées négatives

        # Dessine le texte avec Pillow (compatible arabe)
        draw.text(text_position, predicted_character, font=font, fill=(0, 0, 0))

        # Convertir de nouveau PIL Image en OpenCV BGR
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
