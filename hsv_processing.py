import numpy as np
import cv2 as cv
import datetime

save_time = None

def show(frame):
    cv.imshow('', frame)
    cv.waitKey()
    #cv.destroyAllWindows()
    
def current_time():
    global save_time
    save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    return save_time

def preprocess(action_frame):
    blur = cv.GaussianBlur(action_frame, (3,3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])
    
    mask = cv.inRange(hsv, lower_color, upper_color)
    blur = cv.medianBlur(mask, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    hsv_d = cv.dilate(blur, kernel)

    return hsv_d


def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        origin = frame.copy()


        cv.rectangle(frame, (250, 250),  (450, 450), (0, 0, 255), 2)
        roi = origin[250:450, 250:450]

        cv.imshow("cam", frame)
        cv.imshow("roi", roi)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thre = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        hsv_img = preprocess(roi)

        #cv.imshow("threshold", thre)    

        cv.imshow("hsv", hsv_img)
        if cv.waitKey(30) & 0xFF == ord('s'):
            save_time = current_time()
            cv.imwrite(f"./{save_time}_roi.jpg", roi)
            cv.imwrite(f"./{save_time}_hsv.jpg", hsv_img)
            break
        
main()