"""Receuil de fonctions de test et pour edittage manuel de video"""
import cv2 as cv
import numpy as np
import ball
import decimal
import tkinter as tk
from tkinter import filedialog
from tkinter import LabelFrame
from PIL import Image, ImageTk

global FIRST_VIDEO_PATH
global SECOND_VIDEO_PATH

HUMAN_SZ = 75
ESTIMATE_AIR_DRAG = .85
BATPOS = 505
ERROR_MARGIN = 10
DETECT_ZONE = 135
MARGIN_RED_LEFT = 20
MARGIN_RED_RIGHT = 50

def loadframe(name:str):
    frames = list()
    cap = cv.VideoCapture(name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return frames
        frames.append(frame)
    raise ConnectionAbortedError

def chooseFrame(frames:list,name:str,fonction=None):  #defille au travers des frames
    index = 0
    nindex = BATPOS
    red = 0
    while True:
        frame = frames[index]

        og = frame
        #if not fonction is None: frame = fonction(frame,frames[index+1])

        frame = mask_speed(frame,frames[index+1])
        cv.imshow(name, frame)

        for i in range(0,len(og)):
            og[i][red] = np.array([0,0,255])
        for i in range(0,len(og[0])):
            og[nindex][i] = np.array([0,255,0])
        cv.imshow("og", og)
        key = cv.waitKey(1)

        if key == ord("q"):
            return(index)
        elif key == ord("a"):
            index -= 1
            print(index)
        elif key == ord("d"):
            index += 1
            print(index)
        elif key == ord("c"):
            nindex -= 1
            print(nindex)
        elif key == ord("z"):
            nindex += 1
            print(nindex)
        elif key == ord("r"):
            red = getRd_x(frame)
            print(red)


def write_video(nom:str,frames:list,fps,low_index=0,high_index=None,format='mp4v',sz=None):
    if high_index==None: high_index = len(frames)
    if sz == None: sz = (len(frames[0]),len(frames[0][0]))

    forma = cv.VideoWriter.fourcc(format[0],format[1],format[2],format[3])
    out = cv.VideoWriter(nom, forma, fps, sz)
    for i in range(low_index, high_index):
        out.write(frames[i])
    out.release()

def mask_speed(frame1,frame2):
    # valeur de gris borner
    lower = np.array([100, 100, 100])
    upper = np.array([255, 255, 255])
    frame1 = cv.inRange(frame1, lower, upper)
    frame2 = cv.inRange(frame2, lower, upper)

    # detecte la balle par le changement de valeur de pixel
    kernel1 = np.array([[0, -1, -1, 0], [-1, 1, 1, -1], [-1, 1, 1, -1], [0, -1, -1, 0]],np.uint8)
    kernel2 = np.array([[0, -1, 0, ],[-1, 1, -1],[0, -1, 0, ]],np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)

    frame1 = (frame1 / 2) + (frame2 / 2)
    frame1 = cv.inRange(frame1, 100, 250)
    frame1 = cv.morphologyEx(frame1, cv.MORPH_HITMISS,kernel1)
    frame1 = cv.morphologyEx(frame1, cv.MORPH_HITMISS, kernel2)
    frame1 = cv.morphologyEx(frame1, cv.MORPH_DILATE, kernel3)
    return frame1

def detect_side_ball():
    name = FIRST_VIDEO_PATH
    nFrame = 0

    nBall = 0
    confirmed = False
    framedict = {}
    meaby_ball = list()
    #frames = list( ) #debug
    cap = cv.VideoCapture(name)
    ret, frame_old = cap.read()
    to_bat = -1

    if not ret:
        cap.release()
        raise InterruptedError
    this_frames_balls = list()
    while cap.isOpened():

        if confirmed: confirmed = False
        #if nFrame >= 630: # debug
        #   pass
        ret, frame_new = cap.read()
        if not ret:
            cap.release()
            #chooseFrame(frames,"review bad frames") #debug
            return framedict
        redX = getRd_x(frame_new)
        if BATPOS-MARGIN_RED_LEFT >  redX or redX > BATPOS + MARGIN_RED_RIGHT or nFrame < to_bat:
            frame_old = frame_new
            nFrame +=1
            this_frames_balls.clear()
            #frames.append(frame_old) #debug
            continue




        mask = mask_speed(frame_old, frame_new)
        #cv.imshow("fuckoff",mask)
        #cv.waitKey(1)
        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for i in contours:
            M = cv.moments(i)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cx < BATPOS: this_frames_balls.append([cx, cy])

        for Ball in meaby_ball:
            temp = this_frames_balls
            confirmed = Ball.is_this_you(temp,nFrame)
            if confirmed:
                high, low = getHeight(frame_old)
                to_bat = Ball.toBat()
                print("loop: ", to_bat)
                framedict[nBall] = [to_bat, high, low]
                nBall += 1
                meaby_ball.clear()
                break

        if not confirmed:
            for pos in this_frames_balls:
                if pos[0] < DETECT_ZONE:
                    meaby_ball.append(ball.Ball(this_frames_balls, nFrame))
                    print("newball")
                    break
        frame_old = frame_new
        this_frames_balls.clear()
        nFrame += 1
    raise InterruptedError

def getRd_x(frame:cv.Mat):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([155, 25, 25])
    upper = np.array([179, 255, 255])
    frame = cv.inRange(frame, lower, upper)
    contours, _ = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    i = max(contours, key=cv.contourArea)
    M = cv.moments(i)
    cx = None
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
    if cx is None: cx = BATPOS
    return cx

def getHeight(frame:cv.Mat): # retourne deux valeurs pour la grandeur boite
    # Y grows down in opencv
    #
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([155, 25, 25])
    upper = np.array([179, 255, 255])
    frame = cv.inRange(frame, lower, upper)
    contours, _ = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


    contourList = list()
    for i in contours:
        M = cv.moments(i)
        Area = cv.contourArea(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contourList.append([Area,[cx,cy],i])
    contourList.sort(key = lambda x:x[0],reverse = True)

    Shirt = contourList.pop(0)
    topS = (tuple(Shirt[2][Shirt[2][:, :, 1].argmin()][0]))[1]
    botS = (tuple(Shirt[2][Shirt[2][:, :, 1].argmax()][0]))[1]

    maxX = Shirt[1][0] + HUMAN_SZ
    minX = Shirt[1][0] - HUMAN_SZ

    red_zone = list()
    for c in contourList:
        if minX < c[1][0] > maxX:
            if c[1][1] > botS:
                red_zone.append(c)
    red_zone.sort(key = lambda x:x[0])

    while len(red_zone)>4:
        red_zone.pop()
    red_zone.sort(key = lambda x:x[1][1],reverse=True)

    #cv.imshow("debug",frame)

    K_candidate = list()
    for K in red_zone:
        K_candidate.append((tuple(K[2][K[2][:, :, 1].argmin()][0]))[1])
    knee = min(K_candidate)

    G_candidate = list()
    for G in red_zone:
        G_candidate.append((tuple(G[2][G[2][:, :, 1].argmax()][0]))[1])
    ground = max(G_candidate)

    total = ground - topS
    top = decimal.Decimal((((topS+botS)/2)-topS)/total)
    bot = decimal.Decimal((knee-topS)/total)

    return top,bot

def choose_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        return file_path
    
def load_first_video():
    global FIRST_VIDEO_PATH
    file_path = filedialog.askopenfilename()
    if file_path:
        FIRST_VIDEO_PATH = file_path
        button.config(state="disabled")
        buttonTwo.config(state="normal")    

def detect_front_ball():
    global SECOND_VIDEO_PATH
    file_path = filedialog.askopenfilename()
    if file_path:
        SECOND_VIDEO_PATH = file_path
        buttonTwo.config(state="disabled")
    
    cap2 = cv.VideoCapture(SECOND_VIDEO_PATH)

    if not cap2.isOpened():
        print("Error: Unable to open the video.")
        return
    
    dictionary = detect_side_ball()
    image_paths = []  # List des iamges résultats
    
    for possible in dictionary.items():
        cap2.set(cv.CAP_PROP_POS_FRAMES, possible[1][0])

        ret, frame = cap2.read()

        if not ret:
            print("Error: Unable to read the frame.")
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
        if baseball_inside_roi(frame, rect_x, rect_y, rect_width, rect_height):
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
    button.config(state="normal")

    current_index = 0

    def show_image(index):
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


def baseball_inside_roi(frame, rect_x, rect_y, rect_width, rect_height):
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


root = tk.Tk()
root.title("SIF1033 - Phase 2")

window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

frameT = LabelFrame(root, text="Analyseur de Strike Zone")
frameT.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)

video = tk.Label(frameT)
video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

button = tk.Button(frameT, text="Sélectionnez la première vidéo", command=load_first_video, width=20, height=1)
button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

buttonTwo = tk.Button(frameT, text="Sélectionnez la deuxième vidéo", command=detect_front_ball, width=20, height=1)
buttonTwo.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
buttonTwo.config(state="disabled")

frameT.columnconfigure(0, weight=1)

root.mainloop()

#Ncap = "devant.mp4"
#Ncap = "Ecote.mp4"
#dictio = detect_side_ball() 
#print(dictio)
#frames = loadframe(Ncap)
#chooseFrame(frames,"lol",mask_speed)

cv.destroyAllWindows()

