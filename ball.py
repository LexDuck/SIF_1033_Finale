ERROR_MARGIN_PIXEL = 6
BATPOS = 505
DETECT_ZONE = 135
REPEAT_POS_NEEDED = 3
BALL_NEEDED_BEFORE_CHECK = 3
REPEAT_SPEED_NEEDED = 2
MARGIN_RED_LEFT = 20
MARGIN_RED_RIGHT = 50
ESTIMATE_AIR_DRAG = .85


class Ball:

    def __init__(self, positions: list, nFrame):
        self.ball_list = {}
        self.Nframe = list()
        positions = self.checkpositions(positions, flag=True)
        self.ball_list = {0: positions}
        self.Nframe.append(nFrame)
        self.frames_to_bat = None


    def is_this_you(self, positions: list, nFrame):
        positions = self.checkpositions(positions)
        if len(positions) >= 1:
            self.Nframe.append(nFrame)
            self.ball_list[len(self.ball_list)] = positions
        if len(self.ball_list) < BALL_NEEDED_BEFORE_CHECK: return False
        # ball list == [nFrame] [positions] [x,y] == [[x,y],[x,y]]
        # repeat = [[pos][frames,]]

        j_f = i_f = 0
        i_x = j_x = 0
        inner_cond = cond = False
        speed = list()
        for i in range(len(self.ball_list) - 1):
            if cond and inner_cond:
                if j_f == i_f: continue  # dodges div by 0
                temp = int((j_x - i_x) / (j_f - i_f))
                if temp <= 0: continue  # no negative or 0 speed
                speed.append(temp)
            elif cond:
                inner_cond = True  # on a besoin de deux en ligne vrai
            else:
                inner_cond = False
            cond = False

            for pos1 in self.ball_list[i]:
                if cond: break
                max_X = pos1[0] + ERROR_MARGIN_PIXEL
                min_X = pos1[0] - ERROR_MARGIN_PIXEL
                max_Y = pos1[1] + ERROR_MARGIN_PIXEL
                min_Y = pos1[1] - ERROR_MARGIN_PIXEL
                for pos2 in self.ball_list[i + 1]:
                    if (max_X > pos2[0] > min_X) and (max_Y > pos2[1] > min_Y):
                        cond = True
                        if inner_cond:
                            j_x = pos1[0]
                            j_f = self.Nframe[i]

                        else:
                            i_x = pos1[0]
                            i_f = self.Nframe[i]
                        break
        if len(speed) > REPEAT_SPEED_NEEDED:
            suspect = list()
            watchlist = list()
            while True:

                convicted = list()
                for s in range(len(speed) - 1):

                    moy = (speed[s] + speed[s + 1]) / 2
                    maxUp = moy + ERROR_MARGIN_PIXEL
                    minUp = moy - ERROR_MARGIN_PIXEL
                    if maxUp < speed[s] or speed[s] < minUp:
                        if s in suspect:
                            convicted.append(s)
                            suspect.remove(s)
                        else:
                            suspect.append(s)
                    if maxUp < speed[s + 1] or speed[s + 1] < minUp:
                        if s + 1 in suspect:
                            convicted.append(s + 1)
                            suspect.remove(s + 1)
                        else:
                            suspect.append(s + 1)
                if watchlist.sort == suspect.sort:
                    suspect.clear()
                for i in sorted(convicted, reverse=True):
                    del speed[i]
                    for susy in sorted(suspect, reverse=True):
                        if susy > i:
                            susy -= 1

                if len(suspect) == 0 and len(speed) >= 2:
                    break
                elif len(speed) < 2:
                    return False
                watchlist = suspect

            moy_speed = (sum(speed) / len(speed)) * ESTIMATE_AIR_DRAG

            x = i_x
            self.frames_to_bat = i_f
            while x < BATPOS:
                if (BATPOS - x) > moy_speed:
                    self.frames_to_bat += 1
                x += moy_speed
            return True
        return False

    def checkpositions(self, positions: list, flag=False):
        index = len(self.ball_list)
        temp = list()
        maxR = BATPOS - MARGIN_RED_LEFT
        if flag:
            maxX = DETECT_ZONE
            for pos in positions:
                if pos[0] < maxX:
                    temp.append(pos)


        else:
            maxX = 30000
            for ball in self.ball_list[index - 1]:
                if ball[0] < maxX:
                    maxX = ball[0]

            for pos in positions:
                if maxR > pos[0] > maxX:
                    temp.append(pos)
        temp.sort()
        return temp

    def toBat(self):
        return self.frames_to_bat



"""old is this you
 def is_this_you(self, positions: list, nFrame):
        positions = self.checkpositions(positions)
        if len(positions) >= 1:
            self.Nframe.append(nFrame)
            self.ball_list[len(self.ball_list)] = positions
        if len(self.ball_list) < BALL_NEEDED_BEFORE_CHECK: return False
        repeat = list()  # ball list == [nFrame] [positions] [x,y] == [[x,y],[x,y]]
        # repeat = [[pos][frames,]]

        for i in range(len(self.ball_list) - 1):
            for pos1 in self.ball_list[i]:
                max_X = pos1[0] + ERROR_MARGIN_PIXEL
                min_X = pos1[0] - ERROR_MARGIN_PIXEL
                max_Y = pos1[1] + ERROR_MARGIN_PIXEL
                min_Y = pos1[1] - ERROR_MARGIN_PIXEL
                for pos2 in self.ball_list[i + 1]:
                    if (max_X > pos2[0] > min_X) and (max_Y > pos2[1] > min_Y):
                        for rep in repeat:
                            if (max_X > rep[0][0] > min_X) and (max_Y > rep[0][1] > min_Y):
                                rep[1].append(self.Nframe[i])
                                break
                            repeat.append([pos1, [self.Nframe[i]]])

        # repeat = [[pos][frames,]]
        if len(repeat) > REPEAT_POS_NEEDED:
            speed = list()
            cond = False
            inner_cond = False
            j_f = i_f = 0

            for i in range(len(repeat) - 1):
                for j in range(i, len(repeat)):

                    for iframe in repeat[i][1]:
                        if cond and inner_cond:
                            if j_f == i_f: continue  # dodges div by 0
                            temp = int(repeat[j][0][0] - repeat[i][0][0] / (j_f - i_f))
                            if temp < 0: continue  # no negative speed
                            speed.append(temp)
                        elif cond:
                            inner_cond = True  # on a besoin de deux en ligne vrai
                        else:
                            inner_cond = False
                        cond = False
                        for jframe in repeat[j][i]:
                            if iframe == jframe:
                                i_f = iframe
                                j_f = jframe
                                cond = True
                                break

            if len(speed) >= REPEAT_SPEED_NEEDED:

                moy_speed = sum(speed) / len(speed)
                x = repeat[0][0][0]
                self.frames_to_bat = self.Nframe[0]
                while x < BATPOS:
                    if (BATPOS - x) < moy_speed:
                        self.frames_to_bat += 1
                    x += moy_speed
                return True
        return False


"""


""" old logic for speed
speed = list()
            for i in range(len(repeat)-1):
                for j in range(i,len(repeat)):
                    if((repeat[j][1] == repeat[i][1])): continue # dodges div by 0
                    temp = int(repeat[j][0][0]-repeat[i][0][0]/(repeat[j][1]-repeat[i][1]))
                    if temp < 0: continue #no negative speed
                    speed.append(temp)
            count = 0
            for i in range(len(speed)):
                if count >= REPEAT_SPEED_NEEDED:
                    x = repeat[0][0][0]
                    self.frames_to_bat = self.Nframe[0]
                    while x < BATPOS:
                        if (BATPOS - x) < speed[i-1]:
                            self.frames_to_bat += 1
                        x += speed[i-1]
                    return True
                count = 0
                max_S = speed[i] + ERROR_MARGIN_PIXEL
                min_S = speed[i] - ERROR_MARGIN_PIXEL

                for j in range(i, len(speed)):
                    if min_S < speed[j] < max_S:
                        count +=1


"""

"""
    def is_this_you(self,positions:list, nFrame):

        if len(self.ball_list) < 3:
            positions = self.checkpositions(positions)
            if len(positions)>=1:
                self.Nframe.append(nFrame)
                self.ball_list[len(self.ball_list)] = positions
            return False

        for i in range(len(self.ball_list[0])):
            for j in range(len(self.ball_list[1])):
                if self.ball_list[0][i][0]-ERROR_MARGIN_PIXEL > self.ball_list[1][j][0]:
                    continue
                speed = (self.ball_list[1][j][0] - self.ball_list[0][i][0])/(self.Nframe[1]-self.Nframe[0])
                path = [self.ball_list[0][i], self.ball_list[1][j]]
                cond, path = self.recursion(2,speed,path)
                if cond:
                    x = path[0][0]
                    self.frames_to_bat = self.Nframe[0]
                    speed = int(speed)
                    while x < BATPOS:
                        if (BATPOS - x) < speed:
                            self.frames_to_bat +=1
                        x += speed
                    return True
        positions = self.checkpositions(positions)
        if len(positions) >= 1:
            self.Nframe.append(nFrame)
            self.ball_list[len(self.ball_list)] = positions
        return False



    def recursion(self,index,speed,path):
        if len(self.ball_list) <= index:
            return True, path
        for i in range(len(self.ball_list[index])):
            if path[index-1][0] - ERROR_MARGIN_PIXEL > self.ball_list[index][i][0]:
                continue
            speed_temp = (self.ball_list[index][i][0] - path[index - 1][0])/(self.Nframe[index-1]-self.Nframe[index])
            if speed - ERROR_MARGIN_PIXEL <= speed_temp <= speed + ERROR_MARGIN_PIXEL:
                pathcpy = path
                pathcpy.append(self.ball_list[index][i][0])
                cond, pathcpy = self.recursion(index+1,speed,pathcpy)

                if cond:
                    return True, pathcpy


        return False, path
"""
