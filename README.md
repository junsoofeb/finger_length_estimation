# finger_length_estimation


***

## 1. 개요

OpenCV와 Chainer_Realtime_Multi-Person_Pose_Estimation을 이용한 손가락 길이 예측

## 2. 구현 환경

1. window10 Home
2. Python 3.6
3. OpenCV-python library 4.1.0
4. Numpy 1.16


## 3. 동작 과정

1. 사용자가 카메라의 위치를 지정하기 위해 첫 프레임을 촬영하고, 추출할 roi의 좌상단 우하단을 마우스로 클릭하여 추출.  
### (roi의 width, height는 25cm로 가정.)  
(find_square()함수를 이용하면 자동으로 정사각형 모양 roi를 찾지만 정확도를 올리기 위해서 demo버전에서는 마우스 이벤트를 사용함)  

2. 추출한 roi에서 Chainer_Realtime_Multi-Person_Pose_Estimation을 이용하여 손의 뼈대 좌표를 측정.  
<https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/README.md>

3. 추출한 좌표 정보를 이용하여 직접 손가락과 손가락 사이의 골짜기 부분의 좌표를 계산.

4. 계산된 결과로 손가락 길이와 손바닥 길이를 예측.  
   roi(25cm X 25cm)를 500 X 500 픽셀로 resize했기 때문에 1 pixel == 0.05 cm



## 4. 결과

![ROI](https://user-images.githubusercontent.com/46870741/68008955-e0b08d80-fcc3-11e9-97f7-3404243c44b8.jpg)


![1](https://user-images.githubusercontent.com/46870741/68023059-875b5500-fce9-11e9-9556-28b44198c5fe.png)


![hand_result](https://user-images.githubusercontent.com/46870741/68008985-f625b780-fcc3-11e9-8ce3-5ce62da79fef.jpg)
