import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO
#------------- Add library ------------#
#from matplotlib import plt # render histograms
from queue import Queue
#--------------------------------------#

def tuple3(val1, val2, val3):
    return np.array([val1, val2, val3], dtype="uint8")

lowerRangeWhite = tuple3(0, 100, 0)
upperRangeWhite = tuple3(255, 255, 255)

## mutate image
def BFSAndSetGray(image, row, col, color = 150):
    h = image.shape[0]
    w = image.shape[1]
    
    queue = Queue() 
    image[row, col] = color
    queue.put((row, col))

    count = 0
    sumR = 0
    sumC = 0
    aim = 0

    while not queue.empty():
        r, c = queue.get()

        count += 1
        sumR += r
        sumC += c
        aim += ((h-r)**2)*(c-w/2)

        R = r+1
        C = c
        if R < h and image[R, C] == 0:
            image[R, C] = color
            queue.put((R, C))
        R = r-1
        C = c
        if R >= 0 and image[R, C] == 0:
            image[R, C] = color
            queue.put((R, C))
        R = r
        C = c+1
        if C < w and image[R, C] == 0:
            image[R, C] = color
            queue.put((R, C))
        R = r
        C = c-1
        if C >= 0 and image[R, C] == 0:
            image[R, C] = color
            queue.put((R, C))

    return count, sumR, sumC, aim

def TopGrayPixel(image, color=150):
    h = image.shape[0]
    w = image.shape[1]

    for r in range(h):
        for c in range(w):
            if image[r, c] == color:
                return r, c

    # so that the car won't turn anywhere :)
    return h, w/2

lowerRangeRoad = tuple3(0, 50, 0)
upperRangeRoad = tuple3(255, 255, 200)
def RoadDetect(image):
    # sth, result = cv2.threshold(image, (200, 200, 200), 255, cv2.THRESH_BINARY)
    # result = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result = cv2.GaussianBlur(result, (3, 3), cv2.BORDER_DEFAULT)
    result = cv2.inRange(result, lowerRangeWhite, upperRangeWhite)

    # img[row, col] :)
    h = result.shape[0]
    w = result.shape[1]
    
    count = 0
    sumR = 0
    sumC = 0
    aim = 0

    for i in range(w):
        if result[h-1, i] == 0:
            tcount, tsumR, tsumC, taim = BFSAndSetGray(result, h-1, i)
            count += tcount
            sumR += tsumR
            sumC += tsumC
            aim += taim
    
    return result, count, sumR, sumC, aim

def SolveWithRoadDetect(speed, angle, image):
    NERVOUS_SPEED = 20
    GOGOGO_SPEED = 30

    roadimg, count, sumR, sumC, aim = RoadDetect(image)
    h = roadimg.shape[0]
    w = roadimg.shape[1]

    sendBack_angle = 0
    sendBack_Speed = -speed

    if count <= 100:
        sendBack_Speed += NERVOUS_SPEED
    else:
        avgC = sumC / count

        sendBack_angle += (avgC - w/2) / 50

        furthestR, furthestC = TopGrayPixel(roadimg)
        sendBack_angle += (furthestC - w/2) / 30

        """if aim > 0:
            sendBack_angle += 2 #int(avgC - w/2) / 2
        else:
            sendBack_angle -= 2"""
            
        sendBack_Speed += GOGOGO_SPEED
        sendBack_Speed -= abs(sendBack_angle)*3 # go slower if the car is turning
        
    return sendBack_Speed, sendBack_angle

def Solve(speed, angle, image):
    sendBack_Speed, sendBack_angle = 0, 0 # SolveWithRoadDetect(speed, angle, image)
    return sendBack_Speed, sendBack_angle
    
def ShowTest(speed, angle, image):
    # print(speed, angle)
    # roadimg, cnt, r, c, a = RoadDetect(image)
    # sobelimg = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    cv2.imshow("original", image)
    # cv2.imshow("Road", roadimg)
    # cv2.imshow("Sobel (1,0)", sobelimg)
    # print( sobelimg[1, 1] )
    
#--------------------------------------#

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        steering_angle = 0  #Góc lái hiện tại của xe
        speed = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe
        
        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#
            ShowTest(speed, steering_angle, image);
            sendBack_Speed, sendBack_angle = Solve(speed, steering_angle, image)

            cv2.waitKey(1)
            #------------------------------------------------------------------------------------------------------#
            print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':
    
    #-----------------------------------  Setup  ------------------------------------------#


    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# backup -------------------------------------------- #
"""
def RoadDetect(image):
    # sth, result = cv2.threshold(image, (200, 200, 200), 255, cv2.THRESH_BINARY)
    # result = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result = cv2.inRange(result, lowerRangeWhite, upperRangeWhite)
    return result
"""