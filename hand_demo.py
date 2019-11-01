import cv2 as cv
import time
import numpy as np


protoFile = "./file/pose_deploy.prototxt"
weightsFile = "./file/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)


mouse_points = []

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


def show(frame):
    cv.imshow("", frame)
    cv.waitKey()
    #cv.destroyAllWindows()

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


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_points.append([x, y])
        print("좌표 :",mouse_points)

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
    

def process():
    frame = cv.imread("./ROI.jpg")
    frameCopy = np.copy(frame)
    frameCopy = cv.resize(frameCopy, (500, 500))
    frameCopy2 = cv.resize(frameCopy, (500, 500))
    
    hsv = hsv_version_1(frameCopy)


    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1

    t = time.time()
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    #print("time taken by network : {:.3f}".format(time.time() - t))

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv.resize(probMap, (500, 500))

        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)

        if prob > threshold :
            cv.circle(frameCopy, (int(point[0]), int(point[1])), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            cv.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv.LINE_AA)
            #print(i, "번째 포인트의 x, y 좌표 ({}, {})".format(point[0], point[1]))

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
    #print("Total time taken : {:.3f}".format(time.time() - t))
    dict_finger = findFingerLength(points)
    #for i in dict_finger.keys():
    #    print(i, str(dict_finger[i] * 0.5) + "mm")
    #show(frameCopy)

    return points

def show_hsv():
    cap = cv.VideoCapture(0)
    flag = 0
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        m = hsv_version_1(frame)
        cv.imshow("cam", frame)
        cv.imshow("1", m)
        if cv.waitKey(1) & 0xFF == ord('s'):
            flag = 1
            break
        elif  cv.waitKey(1) & 0xFF == ord('r'):
            flag = 2
            break
    if flag == 1:
        print("hsv이미지가 적당하지 않아서 종료")
        cap.release()
    elif flag == 2:
        init()
        main()


def main():
    frame = cv.imread("./ROI.jpg")
    frame = cv.resize(frame, (500,500))
    target = frame.copy()
    hsv = hsv_version_1(frame)
    canny = cv.Canny(hsv, 50 , 100)

    points = process()
    #print(points)



    end_points = []
    for p in range(len(points)):
        if p != 0 and p % 4 == 0:
            end_points.append(points[p])
    #print(end_points)
    target_points = []

    #for i in range(len(points) - 1):
    #    cv.circle(frame, points[i], 2, (255, 0, 0), -1)


    if points[5][1] > points[9][1]:
        higher_y = points[9][1]
    else :
        higher_y = points[5][1]


    # 4번 5번 포인트의 x좌표사이  --> 첫째 둘째 손가락 사이
    if points[4][1] < points[5][1]:
        min_y = points[4][1]
    else :
        min_y = points[5][1]

    target_x = None
    target_y = min_y

    for x in range(points[5][0], points[4][0]):
        for y in range(min_y, points[2][1]):
            pixel = canny[y,x]
            if pixel == 255 :
                if target_y < y and x < points[2][0]:
                    target_x = x
                    target_y = y


    palm_point = (target_x, target_y)

    #cv.circle(frame, (target_x, target_y), 2, (0, 0, 255), -1)


    # 6번 10번 포인트의 x좌표사이  --> 둘째 셋째 손가락 사이
    if points[6][1] < points[10][1]:
        min_y = points[6][1]
    else :
        min_y = points[10][1]

    target_x = None
    target_y = min_y

    for x in range(points[10][0], points[6][0]):
        for y in range(min_y, higher_y):
            pixel = canny[y,x]
            if pixel == 255 :
                if target_y < y:
                    target_x = x
                    target_y = y

    #cv.circle(frame, (target_x, target_y), 2, (0, 0, 255), -1)

    #손 마디 점찍기 5번6번의 x중점 <둘째>
    mid_x = (points[5][0] + points[6][0]) // 2
    #cv.circle(frame, (mid_x, target_y), 2, (0, 255, 0), -1)
    target_points.append([mid_x, target_y])




    # 10번 14번 포인트의 x좌표사이  --> 셋째 넷째 손가락 사이
    if points[10][1] < points[14][1]:
        min_y = points[10][1]
    else :
        min_y = points[14][1]

    target_x = None
    target_y = min_y

    for x in range(points[14][0], points[10][0]):
        for y in range(min_y, higher_y):
            pixel = canny[y,x]
            if pixel == 255 :
                if target_y < y:
                    target_x = x
                    target_y = y

    #cv.circle(frame, (target_x, target_y), 2, (0, 0, 255), -1)
    #손 마디 점찍기 9번10번의 x중점<셋째>
    mid_x = (points[9][0] + points[10][0]) // 2
    #cv.circle(frame, (mid_x, target_y), 2, (0, 255, 0), -1)
    target_points.append([mid_x, target_y])


    temp_y = target_y


    # 13번 18번 포인트의 x좌표사이  --> 넷째 다섯째
    if points[13][1] < points[18][1]:
        min_y = points[13][1]
    else :
        min_y = points[18][1]

    target_x = None
    target_y = min_y

    pixel = canny[points[18][1], points[18][0]]
    find_x = points[18][0]
    while(pixel != 255):
        find_x += 1
        pixel = canny[points[18][1], find_x]



    for x in range(find_x, points[13][0]):
        for y in range(min_y, points[13][1]):
            pixel = canny[y,x]
            if pixel == 255 :
                if target_y < y:
                    target_x = x
                    target_y = y

    #cv.circle(frame, (target_x, target_y), 2, (0, 0, 255), -1)
    #손 마디 점찍기 17번18번의 x중점 <다섯째>
    mid_x = (points[17][0] + points[18][0]) // 2
    mid_y = (points[17][1] + target_y) // 2
    #cv.circle(frame, (mid_x, mid_y), 2, (0, 255, 0), -1)
    target_points.append([mid_x, mid_y])



    # <넷쨰>
    mid_x = (points[13][0] + points[14][0]) // 2
    mid_y = (temp_y + target_y) // 2
    #cv.circle(frame, (mid_x, mid_y), 2, (0, 255, 0), -1)
    target_points.insert(2,[mid_x, mid_y])
    target_points.insert(0,points[2])


    result = []

    # 손가락
    for l in range(5):
        cv.line(frame, tuple(target_points[l]), end_points[l], (255,255,0),2)
        length = pitaLength(target_points[l], end_points[l])
        result.append(length*0.5)
        length = str(length * 0.5) + 'mm'
        if l == 0:
            cv.putText(frame, length, target_points[l], cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,0,0),2)
        else:
            cv.putText(frame, length, end_points[l], cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,0,0),2)
    
    #손바닥 길이<위>
    cv.line(frame, (palm_point[0], target_points[2][1]), (target_points[4][0], target_points[2][1]), (255,0,255),2)
    cv.line(frame, palm_point, (target_points[4][0],palm_point[1]), (255,0,255),2)
    cv.line(frame, (target_points[2][0], target_points[2][1] ), (target_points[2][0], palm_point[1] ), (255,0,255),2 )
    result.append((palm_point[1] - target_points[2][1]) * 0.5)
    length = str((palm_point[1] - target_points[2][1]) * 0.5) + 'mm'
    cv.putText(frame, length, (target_points[2][0], palm_point[1] - 20 ), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,0,0),2)

    #손바닥 길이<아래>
    cv.line(frame, (palm_point[0],points[0][1]) , (target_points[4][0], points[0][1]), (255,0,255),2)
    cv.line(frame, (target_points[2][0],palm_point[1] ), (target_points[2][0], points[0][1] ), (255,0,255),2 )
    result.append((points[0][1] - palm_point[1] ) * 0.5)    
    length = str((points[0][1] - palm_point[1] ) * 0.5) + 'mm'
    cv.putText(frame, length, (target_points[2][0], points[0][1] - 20), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,0,0),2)
    
    show(frame)
    print(result) # 0~4 : 손가락 길이 5~6:손바닥 길이
    cv.imwrite("./hand_result.jpg", frame)

    #return result

#show_hsv()

main()