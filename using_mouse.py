import cv2

cap_video = cv2.VideoCapture('./YOLOV8_SiamFC/video/test1.mp4')

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
        
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', mouse_callback)

while cap_video.isOpened():
    ret, frame_video = cap_video.read()
    
    if not ret:
        break
    
    if not ret:
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame_video = cv2.resize(frame_video, (video_width, video_height))
    
    output_frame = frame_video[y_offset:y_offset + output_height, x_offset:x_offset + output_width]

    cv2.imshow('Result', output_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap_video.release()
cv2.destroyAllWindows()