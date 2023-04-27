from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
from paddleocr import PaddleOCR
import cv2

yolov7 = YOLOv7()


ocr = PaddleOCR(use_angle_cls=True, lang="arabic")
yolov7.load('best.weights', classes='classes.yaml',
            device='cpu')


video = cv2.VideoCapture('vlc.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output2.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
    print('[!] error opening the video')

print('[+] started reading text on the video...\n')
pbar = tqdm(total=frames_count, unit=' frames',
            dynamic_ncols=True, position=0, leave=True)
texts = {}

try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            # Detect text with PaddleOCR

            detections = yolov7.detect(frame)
            detected_frame = draw(frame, detections)
            output.write(detected_frame)
            pbar.update(1)
            result = ocr.ocr(frame)
            for item in result:
                print("item", item)
        else:
            break

            # Convert PaddleOCR detection results to the same format as YOLOv7
            # detection = {
            #     'class': 'license-plate',
            #     'confidence': item[1],
            #     'x': int(item[0][0][0]),
            #     'y': int(item[0][0][1]),
            #     'width': int(item[0][2][0] - item[0][0][0]),
            #     'height': int(item[0][2][1] - item[0][0][1]),
            #     'color': '#e6281e',
            #     'id': len(detections) + 1
            # }
            # detections.append(detection)

            # text = item[0][1]
            # detection_id = detection['id']
            # if len(text) > 0:
            #     if detection_id not in texts:
            #         texts[detection_id] = {
            #             'most_frequent': {
            #                 'value': '',
            #                 'count': 0
            #             },
            #             'all': {}
            #         }

            #     if text not in texts[detection_id]['all']:
            #         texts[detection_id]['all'][text] = 0

            #     texts[detection_id]['all'][text] += 1

            #     if texts[detection_id]['all'][text] > texts[detection_id]['most_frequent']['count']:
            #         texts[detection_id]['most_frequent']['value'] = text
            #         texts[detection_id]['most_frequent']['count'] = texts[detection_id]['all'][text]

            # if detection_id in texts:
            #     detection['text'] = texts[detection_id]['most_frequent']['value']

            # detected_frame = draw(frame, detections)
            # output.write(detected_frame)
            pbar.update(1)
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()
