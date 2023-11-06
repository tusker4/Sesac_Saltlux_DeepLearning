# yolo v3 keras 기반 모델을 이용한 객체 탐지
    - 이미지 입력후 객체 탐지
    - 카메라를 통한 실시간 영상에서 객체 탐지
        - 라즈베리파이 같은 장비에 장착
        - 일반 데탑용 가능
    - 웹상에서 카메라를 통한 실시간 영상에서 객체 탐지(생략)
        - 웹브라이저 진행,
        - 웹 앱일경우는 스마트폰에서 진행

# 설치
    - opencv 설치
    - 가상환경기반
        -  pip install opencv-python

# yolov3 다운로드
- 가중치 파일  (yolov3.weights)
    - https://pjreddie.com/media/files/yolov3.weights

- 모델구성정보 (yolov3.cfg)
    - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    
- 초기모델의 학습시 사용한  class 값 ()
    - https://github.com/pjreddie/darknet/blob/master/data/coco.names
    - coco_labels.txt 저장

# 학습 방식
    - 제로샷
    - 카메라로 객체를 탐지하는 것 목적

# 용도
    - 정지 이미지용
        - app.py
    - 카메라를 이용한 영상 처리용
        - app.live.py

# 차후 확장
    - 웹용
        - javascript <-> flask(or fastapi) 등과 연동