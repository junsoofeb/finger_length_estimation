import cv2 as cv
import time
import numpy as np


protoFile = "./file/pose_deploy.prototxt"
weightsFile = "./file/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

mouse_points = []

def show(frame):
    cv.imshow("", frame)
    cv.waitKey()
    #cv.destroyAllWindows()



def pitaLength(point1, point2):
    x = (point2[0] - point1[0])**2
    y = (point2[1] - point1[1])**2
    length = round(np.sqrt(x+y))
    return length

def vertLength(point1, point2):
    y = point2[1] - point1[1]
    y_acc = abs(round(y*1.5))
    
    return y_acc

def findFingerLength(points):
    dict_finger = {}
    dict_finger['first'] = pitaLength(points[4], points[2])
    dict_finger['second'] = vertLength(points[8], points[6])
    dict_finger['third'] = vertLength(points[12], points[10])
    dict_finger['four'] = vertLength(points[16], points[14])
    dict_finger['five'] = round(vertLength(points[20], points[18]) / 1.5 * 1.4)

    return dict_finger


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_points.append([x, y])
        print("좌표 :",mouse_points)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def init():
    cap = cv.VideoCapture(0)
    print("s를 눌러서 첫프레임 촬영")
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        #frame = rotateImage(frame, 180)
        cv.imshow("CAM", frame)
        if cv.waitKey(30) & 0xFF == ord('s'):
            cv.imwrite("./set.jpg", frame)
            cv.destroyAllWindows()
            break
    
    img = cv.imread("./set.jpg")
    print("추출할 roi의 좌상단 우하단을 클릭")

    cv.namedWindow('set_image')
    cv.setMouseCallback('set_image', on_mouse)
    cv.imshow("set_image", img)
    while True:
        if len(mouse_points) == 2:
            x = mouse_points[0][0]
            y = mouse_points[0][1]
            ROI = img[y:mouse_points[1][1], x:mouse_points[1][0]]
            break 
        cv.waitKey(30)
    cv.destroyAllWindows()
    print("roi 추출 완료")    
    print("s를 눌러서 손 촬영")
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        ROI = frame[y:mouse_points[1][1], x:mouse_points[1][0]]
        cv.imshow("ROI", ROI)
        if cv.waitKey(30) & 0xFF == ord('s'):
            cv.imwrite("./ROI.jpg", ROI)
            cv.destroyAllWindows()
            break


def hsv_version_1(action_frame):
    blur = cv.GaussianBlur(action_frame, (3,3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    '''
    [0, 58, 50] lower bound skin HSV
    [30, 255, 255] upper bound skin HSV
    '''
    mask = cv.inRange(hsv, lower_color, upper_color)
    blur = cv.medianBlur(mask, 5)

    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8)) 너무 손 두껍게 나와서 3으로 줄임
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    hsv = cv.dilate(blur, kernel)

    return hsv


def hsv_version_2(image, minCr = 128, maxCr=170, minCb = 73, maxCb = 158):
    YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    planes = cv.split(YCrCb)
    nr = image.shape[0]
    nc = image.shape[1]
    mask = np.zeros_like(image)
    
    for i in range(nr):
        CrPlane = planes[1][i]
        CbPlane = planes[2][i]
        for j in range(nc):
            if( (minCr < CrPlane[j]) and (CrPlane[j] <maxCr) and (minCb < CbPlane[j]) and (CbPlane[j] < maxCb) ):
                mask[i, j] = 255 
                
    return mask

def find_square():
    roi = None
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        copy = frame.copy()
        r = frame.copy()
        blur = cv.bilateralFilter(frame, 5, 150, 150)
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 50, 150)
        cnts, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        index_list = []
        #pos_list = []
        
        for i in range(len(cnts)):
            cnt = cnts[i]
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)        
    
            min_x = min(box[:, 0])
            max_x = max(box[:, 0])
            min_y = min(box[:, 1])
            max_y = max(box[:, 1])
    
            garo = max_x - min_x
            sero = max_y - min_y
            
            try:
                ratio = garo / sero
            except:
                continue
            
            if garo >= 125 and sero >= 125 and ratio >= 0.75 and ratio <= 1.25: # 0.85정도 나옴
                #print(ratio)
                index_list.append(i)
                #pos_list.append([min_y, max_y, min_x, max_x])    
                try:
                    roi = r[min_y : max_y , min_x  : max_x ]
                    #print("length :", str(garo * 0.05) + 'cm')
                except:
                    continue
        try:
            cv.imshow("roi", roi)
            cv.waitKey(30)
        except:
            continue
        for ii in range(len(index_list)):
            cnt = cnts[ii]
            cv.drawContours(copy, [cnt], 0, (0, 0, 255) , 4)
        
        cv.imshow("target", copy)
        if cv.waitKey(30) & 0xFF == ord('s'):

            cv.imwrite("./img/target.jpg", roi)
            break


def process():
    frame = cv.imread("./ROI.jpg")
    #frame = cv.imread("./img/test6.jpg")
    frameCopy = np.copy(frame)
    frameCopy = cv.resize(frameCopy, (500, 500))
    frameCopy2 = cv.resize(frameCopy, (500, 500))
    
    hsv = hsv_version_1(frameCopy)


    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    # inWidth = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv.resize(probMap, (500, 500))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)

        if prob > threshold :
            cv.circle(frameCopy, (int(point[0]), int(point[1])), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            cv.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv.LINE_AA)
            print(i, "번째 포인트의 x, y 좌표 ({}, {})".format(point[0], point[1]))

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv.line(frameCopy2, points[partA], points[partB], (0, 255, 255), 2)
            cv.circle(frameCopy2, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
            cv.circle(frameCopy2, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)

    cv.imwrite('Output-Keypoints.jpg', frameCopy)
    cv.imwrite('Output-Skeleton.jpg', frameCopy2)
    print("Total time taken : {:.3f}".format(time.time() - t))
    dict_finger = findFingerLength(points)
    for i in dict_finger.keys():
        print(i, str(dict_finger[i] * 0.5) + "mm")
    show(frameCopy)

#find_square()

init()
process()
