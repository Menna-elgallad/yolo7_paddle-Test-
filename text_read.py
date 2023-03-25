from PIL import Image, ImageDraw
from paddleocr import PaddleOCR, draw_ocr

# img = "./images/te.jpg"


# def read_arabic_image(img):
#     ocr = PaddleOCR(lang="ar")
#     result = ocr.ocr(img)
#     # the image must be one dimention change this
#     # Note: change this accourding to your image shape
#     read, _ = result[0][0][1]
#     return result


# The model file will be downloaded automatically when executed for the first time
ocr = PaddleOCR(use_angle_cls=True, lang="arabic")
img_path = "./images/te.jpg"
result = ocr.ocr(img_path, cls=True)
# Recognition and detection can be performed separately through parameter control
# result = ocr.ocr(img_path, det=False)  Only perform recognition
# result = ocr.ocr(img_path, rec=False)  Only perform detection
# Print detection frame and recognition result
for line in result:
    print(line)

# Visualization
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
print("scores", scores)
im_show = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
