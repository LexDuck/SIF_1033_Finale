"""Receuil de fonctions de test et pour edittage manuel de video"""
import cv2 as cv
import numpy as np
def loadframe(frames:list,cap:cv.VideoCapture):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return frames
        frames.append(frame)
    raise ConnectionAbortedError

def chooseFrame(frames:list,name:str):  #defille au travers des frames
    index = 0
    while True:
        frame = frames[index]
        og = frame
        frame2 = frames[index+1]
        lower = np.array([155, 155, 155])
        upper = np.array([255, 255, 255])
        ground = np.array([0,0,0])
        frame = cv.inRange(frame, lower, upper)
        frame2 = cv.inRange(frame2, lower, upper)
        """space to fuck with frame for debugging / testing shit"""
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #frame = cv.GaussianBlur(frame, (5, 5), 0)
        #frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        #frame2 = cv.GaussianBlur(frame2, (3, 3), 0)
        frame3 = frame - frame2



        cv.imshow(name, frame)
        #cv.imshow("4", frame4)
        cv.imshow("3", frame3)
        cv.imshow("og",og)
        key = cv.waitKey(1)

        if key == ord("q"):
            return index
        elif key == ord("a"):
            index -= 1
        elif key == ord("d"):
            index += 1

def write_video(nom:str,frames:list,fps,low_index=0,high_index=None,format="mpv4",sz=None):
    if high_index==None: high_index = len(frames)
    if sz == None: sz = (len(frames[0]),len(frames[0][0]))

    format = cv.VideoWriter.fourcc(*format)
    out = cv.VideoWriter(nom, format, fps, sz)
    for i in range(low_index, high_index):
        out.write(frames[i])
    out.release()



#Ncap = "devant.mp4"
Ncap = "Ecote.mp4"
frames = list()
cap = cv.VideoCapture(Ncap)
frames = loadframe(frames,cap)
index = chooseFrame(frames,Ncap)


cv.destroyAllWindows()

