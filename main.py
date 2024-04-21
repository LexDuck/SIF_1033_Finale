from FrontView import *
import tkinter as tk
from tkinter import filedialog
from tkinter import LabelFrame

global FIRST_VIDEO_PATH
global SECOND_VIDEO_PATH
    
def load_first_video(): #Permet à l'utilisateur de choisir la première vidéo, dans notre cas la vue de côté
    global FIRST_VIDEO_PATH
    file_path = filedialog.askopenfilename()
    if file_path:
        FIRST_VIDEO_PATH = file_path
        if SECOND_VIDEO_PATH:
            buttonStart.config(state="normal")
    
        
def load_second_video(): #Permet à l'utilisateur de choisir la deuxième vidéo, dans notre cas la vue de face
    global SECOND_VIDEO_PATH
    file_path = filedialog.askopenfilename()
    if file_path:
        SECOND_VIDEO_PATH = file_path
        if FIRST_VIDEO_PATH:
            buttonStart.config(state="normal")
        
        
def main():
    detect_front_ball(FIRST_VIDEO_PATH, SECOND_VIDEO_PATH)

root = tk.Tk()  #Instance principale pour l'interface
root.title("SIF1033 - Phase 2")

window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

frameT = LabelFrame(root, text="Analyseur de Strike Zone") #Contour dans l'interface
frameT.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)

video = tk.Label(frameT) #Le label pour l'affichage de la vidéo
video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

button = tk.Button(frameT, text="Sélectionnez la première vidéo", command=load_first_video, width=20, height=1) #Bouton pour la sélection de vidéo
button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

buttonTwo = tk.Button(frameT, text="Sélectionnez la deuxième vidéo", command=load_second_video, width=20, height=1) #Bouton pour la sélection de vidéo
buttonTwo.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

buttonStart = tk.Button(frameT, text="Débuter", command=main, width=20, height=1) #Bouton pour le lancement du traitement
buttonStart.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
buttonStart.config(state="disabled")

frameT.columnconfigure(0, weight=1)

root.mainloop()