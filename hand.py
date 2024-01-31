import cv2
import mediapipe as mp
import numpy as np
import math
import time
from google.protobuf.json_format import MessageToDict

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        # (x1*x2 + y1*y2)/sqrt(sqrt(x1^2 + y1^2)*sqrt(x2^2 + y2^2))
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_):
    '''
        获取对应手指相关向量的二维角度,计算angle1
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[5][0])-int(hand_[6][0])),(int(hand_[5][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[9][0])- int(hand_[10][0])),(int(hand_[9][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[13][0])- int(hand_[14][0])),(int(hand_[13][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[17][0])- int(hand_[18][0])),(int(hand_[17][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def hand_angle_(hand_):
    '''
        获取对应手指相关向量的二维角度,计算angle2
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[2][0])- int(hand_[3][0])),(int(hand_[2][1])- int(hand_[3][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[5][0])),(int(hand_[0][1])- int(hand_[5][1]))),
        ((int(hand_[5][0])- int(hand_[6][0])),(int(hand_[5][1])- int(hand_[6][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[9][0])),(int(hand_[0][1])- int(hand_[9][1]))),
        ((int(hand_[9][0])- int(hand_[10][0])),(int(hand_[9][1])- int(hand_[10][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[13][0])),(int(hand_[0][1])- int(hand_[13][1]))),
        ((int(hand_[13][0])- int(hand_[14][0])),(int(hand_[13][1])- int(hand_[14][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[17][0])),(int(hand_[0][1])- int(hand_[17][1]))),
        ((int(hand_[17][0])- int(hand_[18][0])),(int(hand_[17][1])- int(hand_[18][1])))
        )
    angle_list.append(angle_)
    return angle_list

def classify_angle(angle1, angle2, cls_num=7):
    '''
        按照规则对手指角度进行分类
    '''
    if angle1 < 0 or angle1 > 180:
        return False  
    if angle2 < 0 or angle2 > 180:
        return False
    
    # cls_num分级以angle1角度判断
    interval = 180 / cls_num
    categories = [i for i in range(1, cls_num+1)]
    category_index = int(angle1 // interval)
    try:
        category = categories[category_index]
    except:
        category = 7
    
    # 达到cls_num之后以angle2角度判断是否完全握拳(cls_num+1级)
    if category==cls_num:
        # print('angle1',angle1)
        # print('angle2',angle2)
        condi_ = int(angle2 // 65)
        category += condi_
    
    return category

def classify_hands_angle(angle1_list, angle2_list):
    '''
        返回手指分类结果，包括一个列表和一个字典
    '''
    clslis = []
    key = ['thumb', 'index', 'middle', 'ring', 'pink']
    dic = {}
    for idx, angle in enumerate(angle1_list):
        cls_ = classify_angle(angle, angle2_list[idx])
        cls_ = cls_ if cls_ else 0
        clslis.append(cls_)
        dic[key[idx]] = cls_
    return clslis, dic

def draw_dict_on_image(frame, data_dict):
    '''
        可视化显示，在图片上显示字典数据
    '''
    # 设置文本参数
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # 白色
    font_thickness = 1

    # 获取文本框的大小
    line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + 10
    text_x = 10
    text_y = frame.shape[0] - line_height - 10

    result_frame = frame.copy()

    # 逐行打印文本
    for key, value in data_dict.items():
        text = f'{key}: {value}'
        cv2.putText(result_frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        text_y -= line_height  # 调整下一行文本的位置

    return result_frame
    
def draw_landmarks_on_image(image, results):
    '''
        可视化显示，将预测的结果显示在图片上
    '''
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    hand_landmarks_list = list(results.multi_hand_landmarks)
    handedness_list = list(results.multi_handedness)
    annotated_image = np.copy(image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = MessageToDict(handedness_list[idx])
        # print(handedness)
        
        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness['classification'][0]['label']}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def main():
    # 设置模型参数
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,            # 实时模式，非静态
            max_num_hands=1,                    # 最大捕捉手的数目，这里设置为单手
            min_detection_confidence=0.75,      # 关键点检测置信度阈值
            min_tracking_confidence=0.75)       # 关键点追踪置信度阈值
    
    # 设置保存视频的格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

    # 设置摄像头设备号，0为默认摄像头
    cap = cv2.VideoCapture(0)
    
    fps_ = 0.0   # 帧率,对每帧的运行时间进行计算
    
    # 初始化计数器和时间戳
    frame_count = 0
    start_time = time.time()
    
    while True:
        t1 = time.time()
        ret,frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 因为摄像头是镜像的，所以将摄像头水平翻转，不是镜像的可以不翻转
            frame= cv2.flip(frame,1)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    # 将结果保存为列表
                    for i in range(21):
                        x = hand_landmarks.landmark[i].x*frame.shape[1]
                        y = hand_landmarks.landmark[i].y*frame.shape[0]
                        hand_points.append((x,y))
                    if hand_points:
                        angle1_list = hand_angle(hand_points)
                        angle2_list = hand_angle_(hand_points)
                        res, dic = classify_hands_angle(angle1_list, angle2_list) 
                        frame = draw_dict_on_image(frame, dic)
                        # 结果 列表的形式呈现分类结果，按顺序分别为['thumb', 'index', 'middle', 'ring', 'pink']
                        print(res)   # 打印输出,这里可以替换为其他操作
                
                # fps的计算
                fps_ = (fps_ + (1. / (time.time() - t1))) / 2  #此处的time.time()就是检测完这张图片的结束时间,除以2是为了和之前的fps求一个平均
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= 1.0:  # 每隔1秒更新一次
                    fps = frame_count / elapsed_time
                    print("FPS:", round(fps, 2))  # 打印输出帧率
                    frame_count = 0
                    start_time = current_time
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 可视化处理结果
                frame = draw_landmarks_on_image(frame, results)
            else:
                fps_ = 0.0  # 没有检测出来的帧fps算作0
            
            # 可视化
            cv2.imshow('Hands', frame)
            out.write(frame)
            # 按下esc退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    out.release()

if __name__=='__main__':
    main()