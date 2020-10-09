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

# import utils

#--------------------------------------#

#Global variable
MAX_SPEED = 30
MAX_ANGLE = 25
# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

#init our model and image array as empty
model = None
prev_image_array = None

IMAGE_WIDTH = 320;
IMAGE_HEIGHT = 180;
oldNegTheta = 2 * np.pi * 7 / 8
oldPosTheta = 2 * np.pi * 1 / 8
oldNegRho = IMAGE_WIDTH / (2*np.sqrt(2))
oldPosRho = IMAGE_WIDTH / (2*np.sqrt(2))

TESTMODE = False

# number of consecutive frame that the car velocity is less than MinSafeSpeed (too weak)
# use to detect slope
MinSafeSpeed = 10
MaxSafeSpeed = 25
slowFrameCnt = 0
fastFrameCnt = 0

def renderLine(dest, theta, rho):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(dest, (x1, y1), (x2, y2), 128, 1)

# This algorithm calculate the average line of 2 lines
# and follow the theta property
def Algo1(negTheta, negRho, posTheta, posRho, y0, angleMult, velocMult, velocBase):
    def calX(theta, rho, y):
        return (rho - y0*np.sin(theta)) / np.cos(theta)

    x1 = calX(negTheta, negRho, y0)
    x2 = calX(posTheta, posRho, y0)

    angle = angleMult * (x1 + x2 - IMAGE_WIDTH)
    veloc = velocBase

    return angle, veloc

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
            * depth_image: ảnh chiều sâu được xe trả về (xét takeDepth = True, ảnh depth sẽ được trả về sau khi 'send_control' được gửi đi)
        
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


            #LINE DETECTION GOES HERE
            #plan: there is none lol

            #region of interest masking
            #cv2.imshow("original", image)

            topInterest = 110
            external_poly = np.array([[[0, 0], [0, topInterest], [320, topInterest], [320, 0]]], dtype=np.int32)
            cv2.fillPoly(image, external_poly, (0, 0, 0))

            #attemp 1: to hsv, filter gray-ish and white-ish color (nvm only white)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_bound_white = (0, 0, 175)
            upper_bound_white = (50, 50, 255)

            # attempt 3: search for road color
            road_color_sample1 = image[179, 160]
            road_color_sample2 = image[120, 160]
            #cv2.line(hsv_image, (40, 0), (40, 200), (0, 0, 0), 3)
            #cv2.line(hsv_image, (120, 0), (120, 200), (0, 0, 0), 3)
            UPPER_BOUND_DIFF = 15
            LOWER_BOUND_DIFF = 15

            lower_road_bound1 = (road_color_sample1[0] - UPPER_BOUND_DIFF, road_color_sample1[1] - UPPER_BOUND_DIFF, road_color_sample1[2] - UPPER_BOUND_DIFF)
            upper_road_bound1 = (road_color_sample1[0] + LOWER_BOUND_DIFF, road_color_sample1[1] + LOWER_BOUND_DIFF, road_color_sample1[2] + LOWER_BOUND_DIFF)
            lower_road_bound2 = (road_color_sample2[0] - UPPER_BOUND_DIFF, road_color_sample2[1] - UPPER_BOUND_DIFF, road_color_sample2[2] - UPPER_BOUND_DIFF)
            upper_road_bound2 = (road_color_sample2[0] + LOWER_BOUND_DIFF, road_color_sample2[1] + LOWER_BOUND_DIFF, road_color_sample2[2] + LOWER_BOUND_DIFF)
            #print(upper_road_bound)
            mask_road1 = cv2.inRange(image, np.float32(lower_road_bound1), np.float32(upper_road_bound1))
            mask_road2 = cv2.inRange(image, np.float32(lower_road_bound2), np.float32(upper_road_bound2))
            filtered1 = cv2.bitwise_and(image, image, mask=mask_road1)
            filtered2 = cv2.bitwise_and(image, image, mask=mask_road2)
            filtered = cv2.addWeighted(filtered1, 0.5, filtered2, 0.5, 0.5)

            hsv_mask = cv2.inRange(hsv_image, lower_bound_white, upper_bound_white)
            hsv_filtered = cv2.bitwise_and(hsv_image, hsv_image, mask=hsv_mask)
            # convert to edge by canny
            hsv_edge = cv2.Canny(hsv_filtered, 50, 150)

            #detech line
            lines = cv2.HoughLines(hsv_edge, 1, np.pi / 180, 50)

            global oldNegRho, oldNegTheta, oldPosRho, oldPosTheta, slowFrameCnt, fastFrameCnt, TESTMODE

            avgTheta = 0;
            avgRho = 0;
            
            negTheta = 0;
            negRho = 0;
            negCnt = 0;

            posTheta = 0;
            posRho = 0;
            posCnt = 0;
            

            lowerTheta = np.pi / 20;
            
            if lines is not None:
                for line in lines:
                    for rho, theta in line:
                        temptheta = theta
                        if theta >= np.pi:
                            temptheta -= np.pi

                        if not (temptheta >= np.pi/2 - lowerTheta and temptheta <= np.pi/2 + lowerTheta):
                            if temptheta < np.pi / 2:
                                posTheta += theta
                                posRho += rho
                                posCnt += 1
                            else:
                                negTheta += theta
                                negRho += rho
                                negCnt += 1

            if negCnt != 0:
                negTheta /= negCnt
                negRho /= negCnt
            else:
                negTheta = oldNegTheta
                negRho = oldNegRho
                
            if posCnt != 0:
                posTheta /= posCnt
                posRho /= posCnt
            else:
                posTheta = oldPosTheta
                posRho = oldPosRho
                
            oldNegTheta = negTheta
            oldNegRho = negRho
            oldPosTheta = posTheta
            oldPosRho = posRho


            # render average line
            renderLine(hsv_edge, negTheta, negRho)
            renderLine(hsv_edge, posTheta, posRho)
            
            
            #################################################################################################### 
            #attemp 2: dont
            # lower_bound_white = (180, 180, 180)
            # upper_bound_white = (255, 255, 255)
            # mask = cv2.inRange(image, lower_bound_white, upper_bound_white)
            # filtered = cv2.bitwise_and(image, image, mask=mask)
            # # convert to edge by canny
            # edge = cv2.Canny(filtered, 50, 150)



            #showing result output to windows (only for visualizing)
            if TESTMODE:
                cv2.imshow("image", image)
                cv2.imshow("hsv", hsv_image)
            
                cv2.imshow("filter", filtered)
                cv2.imshow("edge", hsv_edge)

            #CAR BEHAVIOR GOES HERE
            #plan: if no line is presented, go straight, if L or R line is presented, angle defined by its slope
            #if both are presented, angle defined by 2 lines' intersected point
            
            velocBase = 30
            if negCnt == 0:
                velocBase -= 8
            if posCnt == 0:
                velocBase -= 8
            
            #handle slope
            if speed < MinSafeSpeed:
                slowFrameCnt += 1
            else:
                slowFrameCnt = 0

            if speed > MaxSafeSpeed:
                fastFrameCnt += 1
            else:
                fastFrameCnt = 0

            velocBase += slowFrameCnt * 2
            velocBase -= fastFrameCnt * 2
            velocBase -= speed #re-balance speed

            sendBack_angle, sendBack_Speed = Algo1(negTheta, negRho, posTheta, posRho, IMAGE_HEIGHT - 10, 0.1, 1, velocBase)

            # on turning, you should slow down :)
            sendBack_Speed -= sendBack_angle

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
    