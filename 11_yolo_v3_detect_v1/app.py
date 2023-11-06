import numpy as np
import cv2 as cv
import sys

# 함수 지향적 프로그래밍 방식

def init_yolov3():
    '''
        예측에 필요한 모든 자원 로드
        가중치+모델구조, 정답표(레이블), 예측에 사용하는 레이어층명
    '''
    # 1. 모델 로드 : cv가 가중치와 모델구조 정보를 가지로 로드
    model = cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    
    # 2. 레이어명 획득
    layer_names = model.getLayerNames()
    #print( layer_names )
    #    예측 수행에 사용하는 레이어 이름 인덱스 200 227 254층 사용
    # 인덱스를 1부터 카운트함
    out_layers = [ layer_names[idx-1] for idx in model.getUnconnectedOutLayers() ]
    # ['yolo_82', 'yolo_94', 'yolo_106']
    #print( '출력 담당 레이어명 ', out_layers )
    
    # 3. 정답 데이터 -> yolov3는 80개 객체를 탐지 가능
    with open('coco_labels.txt') as f:
        labels = [ label.strip()  for label in f.readlines() ]
    #print( labels )
    
    # 4. 리턴: 모델, 예측에 사용하는 레이어이름, 레이블(분류표,순서(인덱스번호))
    return model, out_layers, labels

def predict_image(predict_img, model, out_layers):
    # 1. 모델에 이미지 데이터 주입
    model.setInput( predict_img )
    # 2. 모델에 출력층 정보를 세팅해서 예측 수행
    return model.forward( out_layers )

def parse_predict( outputs, labels, img_h, img_w ):
    '''
        예측 결과를 파싱, 바운딩박스좌표, 정확도(신뢰도), 분류번호(값, 이름등) 추출
    '''
    boxes, confs, label_ids = list(), list(), list()
    for idx, output in enumerate(outputs): # 탐지된 객체별(개,자전거,자동차)로 반복
        print( idx, output.shape)
        for info in output: # 객체별로 생성된 바운딩 박스 반복
            # 85(4+1+80), 배열
            # 80개의 분류값(컨피던스값) 추출
            confidence_candidates = info[5:]
            # 80개중 가장 큰값을 가진 인덱스 추출
            max_idx = np.argmax( confidence_candidates )
            # 최대 신뢰도값 획득
            max_confidence = confidence_candidates[ max_idx ]
            # 신뢰도 임계값 기준을 커트라인(0.5)
            # 신뢰도가 50% 이상인 경우만 인정 -> 설정
            if max_confidence > 0.5:
                '''
                    50% 이상 신뢰도를 가진 바운딩 박스 리스트
                    0 (147, 85)        
                        1 bicycle 0.9915967
                    1 (588, 85)        
                        7 truck 0.56994236 
                        7 truck 0.5092606  
                        7 truck 0.59335667 
                        16 dog 0.95524865  
                        16 dog 0.9613249   
                        16 dog 0.98602635  
                        16 dog 0.99229467  
                        16 dog 0.5655222   
                    2 (2352, 85)  
                '''
                #print( max_idx, labels[max_idx], max_confidence)
                # 데이터 담기
                confs.append( max_confidence ) # 신뢰도 
                label_ids.append( max_idx )    # 분류 번호
                c_x, c_y = int(info[0]*img_w), int(info[1]*img_h) # 박스 센터 좌표
                w, h     = int(info[2]*img_w), int(info[3]*img_h) # 박스 너비, 폭
                x, y     = int(c_x-w/2), int(c_y-h/2)
                
                boxes.append( [ x, y, x+w, y+h ] ) # 바운딩박스 정보
                pass # end if
            pass # end for
    pass # end for
    # 샘플 바운딩 박스 정보 추출
    #print(boxes[0], confs[0], label_ids[0], labels[label_ids[0]] )
    # 후보값들 중에서, 가장 최대값 혹은 최적값 추출해서 최종 방스 정보만 추출
    # 노이즈제거, 비최대억제(NMS) 알고리즘을 적용하여 바운딩 박스 최대값 계산
    # (박스후보좌표들, 신뢰도값들, 신뢰도 임계값, NMS임계값(설정값))
    indexs = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4 )
    print( indexs ) # [7 0 3]
    
    # 최종 박스만 정보 추출
    # [ [ x, y, w, h, 신뢰도, 분류아이디 ], [], [] ]
    final_infos = [ boxes[i]+[confs[i]]+[label_ids[i]] for i in range(len(confs)) if i in indexs  ]
    print( final_infos )
    # 객체당 바운딩 박스 1개 추출 완료
    return final_infos
    

def detect_image(model, out_layers, labels):
    '''
        이미지 데이터 준비 -> 입력 혹은 하드코딩
        이미지 데이터를 모델에 주입 -> 예측 수행
    '''
    # 1. 객체 탐지에 대상이 되는 이미지 로드(데이터)
    #    이미지 형식은 BGR 방식
    img_src = cv.imread('dog.jpg')
    if img_src is None: # 이미지가 로드 X
        sys.exit('이미지 파일 누락 혹은 오류로 인해 진행 불가')
    
    # 2. 필요 정보 추출 -> h, w, c(사용않함)
    img_h, img_w, _ = img_src.shape
    #print( img_src.shape, img_h, img_w )
    
    # 3. yolo v3 모델에 입력이 가능한 형태로 이미지 데이터 변환
    # 블롭(blob)형태로 변환
    # size => (224,224) or (448,448) 정사이즈 지원 => 정보손실 존재
    # (0,0,0) => 색상에 대한 평균값
    # swapRB=True : BGR => RGB
    predict_img = cv.dnn.blobFromImage(img_src, 1/256, (224,224), (0,0,0), swapRB=True)
    
    # 4. 예측 수행
    outputs = predict_image(predict_img, model, out_layers)
    # 예측결과 :  (147, 85)
    # 147개 바운딩박스 검출, 85(4+1+80) : 4는 박스좌표, 80은레이블80개에대한 컨피던스, 1은 신뢰도로 사용 X
    print( '예측결과 : ', outputs[0].shape )
    #print( '샘플값 : ', outputs[0][0] )
    
    # 5. 예측 결과를 파싱해서 객체별 바운딩 박스/레이블/정확도 정보 획득
    final_infos = parse_predict( outputs, labels, img_h, img_w )
    
    # 6.  박스 드로잉을 위한 칼라 설정
    #     분류번호별로 별도의 구분되는 색상을 사용
    #     색상 배열 생성, (80, 3)
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    
    # 7. 화면 처리 -> 바운딩 박스 + 분류이름 + 정확도 드로잉
    for info in final_infos:
        # 정보 추출
        x1, y1, x2, y2, confidence, id = info
        # 박스그리기
        cv.rectangle(img_src, (x1, y1), (x2, y2), colors[id], 2)
        # 신뢰도 + 분류값
        text = f'{labels[id]} - {confidence}'
        # 텍스트 그리기
        cv.putText(img_src, text, (x1, y1-20), cv.FONT_HERSHEY_PLAIN, 1.0, colors[id], 2)
        
    # 8 . 이미지 출력
    cv.imshow('Yolo v3 Object Detecting', img_src)
    
    # 9. 쓰레드 대기 -> 무한루프 사용시등 적정한데 사용가능
    cv.waitKey()
    # 10. 윈도우 종료
    cv.destroyAllWindows()
    
        
        
    pass

def main():
    '''
         모든 업무를 절차적으로 진행하는 메인코드, 엔트리포인트
    '''
    # 1. yolo_v3 초기화
    model, out_layers, labels = init_yolov3()
    # 2. 1장의 이미지 객체 탐지 -> 카메라로 탐지
    detect_image(model, out_layers, labels)
    pass

if __name__ == '__main__':
    main()