# 카메라를 이용한 라이브 영상에서 객체 탐지
import numpy as np
import cv2 as cv
import sys

USE_CAMERA_LIVE = True # 라이프 영상 모드

def init_yolov3():
    model = cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [ layer_names[idx-1] for idx in model.getUnconnectedOutLayers() ]
    with open('coco_labels.txt') as f:
        labels = [ label.strip()  for label in f.readlines() ]
    return model, out_layers, labels

def predict_image(predict_img, model, out_layers):
    model.setInput( predict_img )
    return model.forward( out_layers )

def parse_predict( outputs, labels, img_h, img_w ):
    boxes, confs, label_ids = list(), list(), list()
    for idx, output in enumerate(outputs):
        print( idx, output.shape)
        for info in output: 
            confidence_candidates = info[5:]
            max_idx = np.argmax( confidence_candidates )
            max_confidence = confidence_candidates[ max_idx ]
            if max_confidence > 0.5:
                confs.append( max_confidence ) 
                label_ids.append( max_idx )    
                c_x, c_y = int(info[0]*img_w), int(info[1]*img_h) 
                w, h     = int(info[2]*img_w), int(info[3]*img_h) 
                x, y     = int(c_x-w/2), int(c_y-h/2)
                
                boxes.append( [ x, y, x+w, y+h ] ) 
                pass # end if
            pass # end for
    pass # end for
    indexs = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4 )
    print( indexs )
    final_infos = [ boxes[i]+[confs[i]]+[label_ids[i]] for i in range(len(confs)) if i in indexs  ]
    print( final_infos )
    return final_infos
    
def detect_image(model, img_src, out_layers, labels):

    img_h, img_w, _ = img_src.shape
    predict_img = cv.dnn.blobFromImage(img_src, 1/256, (224,224), (0,0,0), swapRB=True)
    outputs = predict_image(predict_img, model, out_layers)
    print( '예측결과 : ', outputs[0].shape )
    final_infos = parse_predict( outputs, labels, img_h, img_w )
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    for info in final_infos:
        x1, y1, x2, y2, confidence, id = info
        cv.rectangle(img_src, (x1, y1), (x2, y2), colors[id], 2)
        text = f'{labels[id]} - {confidence}'
        cv.putText(img_src, text, (x1, y1-20), cv.FONT_HERSHEY_PLAIN, 1.0, colors[id], 2)
    cv.imshow('Yolo v3 Object Detecting', img_src)
    # cv.waitKey()
    # cv.destroyAllWindows()
    pass

def main():
    model, out_layers, labels = init_yolov3()
    # 카메라 모듈이 추가
    if  USE_CAMERA_LIVE:
        capture = cv.VideoCapture(1)
        if capture.isOpened():
            sys.exit('카메라 연결 오류')   
        else: 
            img_src = cv.imread('dog.jpg')
            if img_src is None: 
                sys.exit('이미지 파일 누락 혹은 오류로 인해 진행 불가')
            while True:
                # 캡쳐본 추출
                res, frame = capture.read()
                if not res:
                    sys.exit('프레임 정보 획득 오류')
                
                # yolo예측
                detect_image(model, frame, out_layers, labels)
                
                # 특정키 => z 를 입력하면 종료
                key = cv.waitKey(1)     
                if key == ord('z'):
                    break
        pass
    
    # detect_image(model, out_layers, labels)

if __name__ == '__main__':
    main()