import cv2 as cv
import numpy as np
import ball
import decimal


# J'avais prevus de faire une fonction qui receille les Constante d'un fichier texte pour
# style setting.txt pour pouvoir les modifier plus facilement de en dehors du code mais
# it looks like on l'utilisera pas

HUMAN_SZ = 75
ESTIMATE_AIR_DRAG = .85
BATPOS = 505
ERROR_MARGIN = 10
DETECT_ZONE = 135
MARGIN_RED_LEFT = 20
MARGIN_RED_RIGHT = 50

def loadframe(name:str):
    #fonction prenant une video et capture toute les frame a l'interieur d'une liste
    frames = list()
    cap = cv.VideoCapture(name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return frames
        frames.append(frame)
    raise ConnectionAbortedError

def chooseFrame(name:str,fct=None,fct2=None):
    # defille au travers des frames avec un certain controle
    # essentiellement un outils utiliser lors de la conception

    index = 0
    nindex = 0
    red = 0
    frames = loadframe(name)

    while True:
        frame = frames[index]

        if not fct is None:
            fram = fct2(frame,frames[index+1])
            cv.imshow("frame fcts", fram)

        if not fct2 is None:
            frame2 = fct2(frame,frames[index+1])
            cv.imshow("frame fcts2", frame2)

        for i in range(0,len(frame)):
            frame[i][red] = np.array([0, 0, 255])
        for i in range(0,len(frame[0])):
            frame[nindex][i] = np.array([0, 255, 0])

        cv.imshow(name, frame)
        key = cv.waitKey(1)

        if key == ord("q"):
            return( index )
        elif key == ord("a"):
            index -= 1
            print( index )
        elif key == ord("d"):
            index += 1
            print( index )
        elif key == ord("c"):
            nindex -= 1
            print( nindex )
        elif key == ord("z"):
            nindex += 1
            print( nindex )
        # retourne le centre du spot rouge qui est moi
        elif key == ord("r"):
            red = getRd_x( frame )
            print(red)


def write_video(nom:str,frames:list,fps,low_index=0,high_index=None,format='mp4v',sz=None):
    # Enregistre une video avec certain parrametres
    if high_index==None: high_index = len(frames)
    if sz == None: sz = (len(frames[0]),len(frames[0][0]))

    forma = cv.VideoWriter.fourcc(format[0],format[1],format[2],format[3])
    out = cv.VideoWriter(nom, forma, fps, sz)
    for i in range(low_index, high_index):
        out.write(frames[i])
    out.release()

def mask_speed(frame1,frame2):
    # Fonction prenant deux frame et se servant de la division et l'addition de deux frames succinte

    # Masque pour le gris
    lower = np.array([100, 100, 100])
    upper = np.array([255, 255, 255])
    frame1 = cv.inRange(frame1, lower, upper)
    frame2 = cv.inRange(frame2, lower, upper)

    # detecte la balle par le changement de valeur de pixel
    kernel1 = np.array([[0, -1, -1, 0], [-1, 1, 1, -1], [-1, 1, 1, -1], [0, -1, -1, 0]]).astype(np.uint8)
    kernel2 = np.array([[0, -1, 0, ],[-1, 1, -1],[0, -1, 0, ]]).astype(np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)

    # voir la diffference au niveau des pixels dune frame a l'autre
    frame1 = (frame1 / 2) + (frame2 / 2)
    frame1 = cv.inRange(frame1 ,100 ,250)

    # Filtres morphologiques pour esseyer d'enlever le plus de noise possibles (presque impossible les images suces)
    # HIT_MISS detecte un paterne etablit par le kernel 1 et 2

    frame1 = cv.morphologyEx(frame1, cv.MORPH_HITMISS, kernel1)
    frame1 = cv.morphologyEx(frame1, cv.MORPH_HITMISS, kernel2)
    frame1 = cv.morphologyEx(frame1, cv.MORPH_DILATE, kernel3)

    return frame1

def detect_side_ball(name:str):
    # nerf de la guerre de la detection de l'image de coter

    nFrame = 0                      #Numero de frame
    nBall = 0                       #Nombre de balle detecter et confirmer
    confirmed = False               #Instantiation de la varibale trackant si la balle est confirmer ou non
    framedict = {}                  #Instantiation de la variable de retour ayant le format {nBall: [numero_de_frame,% highbound,%lowbound]} -> plus de detail fcts getHeight
    meaby_ball = list()             #Instantiation de la liste de balle potentielle
    to_bat = -1                     #Variable servant a passer ne pas analyser certaine frame qui n'ont pas de balle puisque la balle a deja ete trouver
    this_frames_balls = list()      #"Balle" dans la frame c'est probablement du bruit pour la pluspart
    #frames = list( ) #debug

    cap = cv.VideoCapture(name)
    ret, frame_old = cap.read()

    if not ret:
        cap.release()
        raise InterruptedError

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


    contourList=list()
    for i in contours:
        M = cv.moments(i)
        Area = cv.contourArea(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contourList.append([Area,[cx,cy],i])
    contourList.sort(key= lambda x: x[0], reverse= True)

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

    while len(red_zone) > 4:
        red_zone.pop()
    red_zone.sort(key=lambda x: x[1][1], reverse=True)

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

    return top, bot

#Ncap = "devant.mp4"
#Ncap = "Ecote.mp4"
#dictio = detect_side_ball(Ncap)
#print(dictio)
#chooseFrame("lol",fct2 = mask_speed)
cv.destroyAllWindows()