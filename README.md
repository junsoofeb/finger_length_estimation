# finger_length_estimation
손가락 길이 예측

## demo.py
(prototype)
***

## 1. 개요

카메라를 사용한 손가락 길이 측정 프로그램

## 2. 구현 환경

1. window10 Home
2. Python 3.6
3. OpenCV-python library 4.1.0
4. Numpy 1.16


## 3. 동작 과정

1. init()함수가 수행된다. s를 눌러서 첫 프레임을 촬영하고, 추출할 roi의 좌상단 우하단을 마우스로 클릭하여 추출한다.  
### (roi는 width, height 모두 25cm인 정사각형으로 제한한다.)  
(find_square()함수를 이용하면 자동으로 roi를 찾지만 정확도를 올리기 위해서 demo버전에서는 마우스 이벤트를 사용함)  

2. 추출한 roi 속에서 s를 눌러 손을 촬영하면 손의 마디 좌표 정보가 추출된다.  
(Chainer_Realtime_Multi-Person_Pose_Estimation 이용)  
<https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/README.md>

3. 추출한 좌표 정보로 손가락 길이를 예측한다. roi를 500 X 500 픽셀로 resize했기 때문에 1 pixel == 0.05 mm가 된다.


## 4. 예시
<1>
![1](https://user-images.githubusercontent.com/46870741/67832785-10299380-fb26-11e9-84ce-adc2602a998c.png)

<2>
![2](https://user-images.githubusercontent.com/46870741/67832786-10c22a00-fb26-11e9-9064-178a33d32f2a.png)

<3>
![3](https://user-images.githubusercontent.com/46870741/67832787-10c22a00-fb26-11e9-9576-02722d4ee5b5.png)

<4>
![4](https://user-images.githubusercontent.com/46870741/67832788-10c22a00-fb26-11e9-8503-5ceabfde8855.png)
