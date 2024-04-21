"""Receuil de fonctions de test et pour edittage manuel de video"""
import cv2 as cv
import numpy as np
from SideView import *
import tkinter as tk
from tkinter import filedialog
from tkinter import LabelFrame
from PIL import Image, ImageTk

def detect_front_ball(first:str, second:str):
    cap2 = cv.VideoCapture(second)

    if not cap2.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo.")
        return
    
    image_paths = []  # List des images résultats
    dictionary = detect_side_ball(first)
    for possible in dictionary.items():
        cap2.set(cv.CAP_PROP_POS_FRAMES, possible[1][0])

        ret, frame = cap2.read()

        if not ret:
            print("Erreur: Impossible d'ouvrir la frame.")
            return
        
        # Calcul de la hauteur
        rect_height_percent = (possible[1][1] - possible[1][2])
        rect_height = int(rect_height_percent * frame.shape[0])

        # Calcul de la longueur
        aspect_ratio = frame.shape[1] / frame.shape[0]
        rect_width = int(aspect_ratio * float(rect_height_percent) * 1.5)  #Le 1.5 est pour la portée du joueur

        # Ajustement de la largeur si trop gros
        if rect_width > frame.shape[1]:
            rect_width = frame.shape[1]

        # Calculer si guache/droite pour la position de frappeur
        if possible[1][0] < BATPOS: 
            rect_x = BATPOS - rect_width
        else:  
            rect_x = BATPOS

        
        rect_y_percent = possible[1][2]
        rect_y = int(rect_y_percent * frame.shape[0])

        top_pixel = int(possible[1][1] * frame.shape[0])
        bot_pixel = int(possible[1][2] * frame.shape[0])

        frame_with_rect = frame.copy()
        # Dessiner le rectangle
        if baseball_inside_rect(frame, rect_x, rect_y, rect_width, rect_height):
            color = (0, 255, 0)  # Vert si dans le rectangle
        else:
            color = (0, 0, 255)  # Rouge si pas dans le rectangle

        cv.rectangle(frame_with_rect, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color, 2)

        # Sauvegarder l'image
        img_path = f"frame_{possible[0]}.png"
        cv.imwrite(img_path, frame_with_rect)
        image_paths.append(img_path)

    cap2.release()

    # Ouvrir l'image viewer
    image_viewer = tk.Toplevel()
    image_viewer.title("Visualisateur des résultats")

    current_index = 0

    def show_image(index): #Les fonctions suivantes servent à l'affichage des images résultats dans un visualisateur
        nonlocal current_index
        current_index = index
        img_path = image_paths[index]
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)

        label.config(image=photo)
        label.image = photo
        label.pack()

    def prev_image():
        nonlocal current_index
        current_index = (current_index - 1) % len(image_paths)
        show_image(current_index)

    def next_image():
        nonlocal current_index
        current_index = (current_index + 1) % len(image_paths)
        show_image(current_index)

    btn_prev = tk.Button(image_viewer, text="<-", command=prev_image)
    btn_prev.pack(side=tk.LEFT)

    btn_next = tk.Button(image_viewer, text="->", command=next_image)
    btn_next.pack(side=tk.RIGHT)

    label = tk.Label(image_viewer)
    show_image(current_index)

    image_viewer.mainloop()


def baseball_inside_rect(frame, rect_x, rect_y, rect_width, rect_height): # Sert à vérifier si la balle est dans la zone de frappe
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (11, 11), 0)

    _, thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        
        center_x = x + w // 2
        center_y = y + h // 2
        
        if rect_x <= center_x <= rect_x + rect_width and rect_y <= center_y <= rect_y + rect_height:
            return True

    return False