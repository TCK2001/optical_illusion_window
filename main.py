#%%
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap_camera = cv2.VideoCapture(0)
cap_video = cv2.VideoCapture('./test1.mp4')

video_width = 1024
video_height = 720

output_width = 960
output_height = 540

x_offset = (video_width - output_width) // 2
y_offset = (video_height - output_height) // 2

def mouse_callback(event, x, y, flags, param):
    global x_offset, y_offset
    if event == cv2.EVENT_MOUSEMOVE:
        # (x - output_width // 2) => -480 ~ +480
        # (y - output_height // 2) => -270 ~ +270
        # video_width - output_width => 64
        # video_height - output_height => 180
        # max를 통해 - 방지하고 일정량이 넘어가면 video_width - output_width 그 전까진 앞 부분.
        x_offset = min(max(0, x - output_width // 2), video_width - output_width)
        y_offset = min(max(0, y - output_height // 2), video_height - output_height)
        print(x_offset, y_offset)
        
cv2.namedWindow('MediaPipe Face Detection')
cv2.setMouseCallback('MediaPipe Face Detection', mouse_callback)

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

while cap_camera.isOpened() and cap_video.isOpened():
    ret_camera, frame_camera = cap_camera.read()
    if not ret_camera:
        break
    
    ret_video, frame_video = cap_video.read()
    
    if not ret_video:
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame_camera_rgb = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_camera_rgb)
    
    frame_video = cv2.resize(frame_video, (video_width, video_height))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            # 얼굴 중심점
            x_center = int(bboxC.xmin * frame_camera.shape[1] + bboxC.width * frame_camera.shape[1] / 2)
            y_center = int(bboxC.ymin * frame_camera.shape[0] + bboxC.height * frame_camera.shape[0] / 2)
            
            # 거리 계산 얼굴이랑 화면
            delta_x = x_center - frame_camera.shape[1] // 2
            delta_y = y_center - frame_camera.shape[0] // 2
            
            sensitivity = 0.3
            x_offset = min(max(0, x_offset + int(delta_x * sensitivity)), video_width - output_width)
            y_offset = min(max(0, y_offset + int(delta_y * sensitivity)), video_height - output_height)
            
            # 얼굴 인식 결과를 카메라 프레임에 그리기 (디버깅용)
            mp_drawing.draw_detection(frame_camera, detection)
    
    output_frame = frame_video[y_offset:y_offset + output_height, x_offset:x_offset + output_width]

    cv2.imshow('MediaPipe Face Detection', output_frame)
    # cv2.imshow('Camera', frame_camera)  # 디버깅을 위해 카메라 화면도 출력
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap_camera.release()
cap_video.release()
cv2.destroyAllWindows()
# %%
