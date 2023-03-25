from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import cv2
import os
import pytesseract
import easyocr
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


yolov7 = YOLOv7()
yolov7.load('best.weights', classes='classes.yaml',
            device='cpu')  # use 'gpu' for CUDA GPU inferenceprint

print("model loaded")


def detection(imgpath):
    image = cv2.imread(imgpath)
    detections = yolov7.detect(image)
    detected_image = draw(image, detections)
    cv2.imwrite('detected_image.jpg', detected_image)
    print(json.dumps(detections, indent=4))
    # print()
    json_data = json.dumps(detections)
    parsed_dict = json.loads(json_data)[0]
    x, y, width, height = parsed_dict["x"], parsed_dict["y"], parsed_dict["width"], parsed_dict["height"]
    # print(x, y, width, hight)
    # image = Image.open(imgpath)
    cropped_image = image[y:y+height, x:x+width]
    # cropped_image = image.crop((x, y, x + width, y + height))

    cv2.imshow("cropped", cropped_image)
    cv2.waitKey(0)
    return cropped_image
    # Save the cropped image
    # cropped_image.save("cropped_image.jpg")


def cropping(plate):
    res = cv2.resize(plate, dsize=(270, 200), interpolation=cv2.INTER_CUBIC)
    colorpart = res[0:80, 0:270]
    cv2.imshow("colorpart", colorpart)
    crop = res[80:200, 0:270]
    return colorpart, crop


def color_extraction(img):
    height, width = img.shape[:2]
    hc = height / 2
    wc = width/2
    color = img[np.int16(hc)][np.int16(wc)]
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    print("Color at center pixel is - Red: {}, Green: {}, Blue: {}".format(red, green, blue))

    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    bars = []
    bars.append(bar)

    img_bar = np.hstack(bars)
    cv2.imshow('Dominant colors', img_bar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(color)
    return (red, green, blue)


def convert_rgb_to_names(rgb_tuple):
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    rred = ["maroon", "darkred", "firebrick", "brown", "crimson", "red"]
    yellow = ["gold", "goldenrod", "khaki", "yellow", "sandybrown", "peru"]
    ggreen = ["yellowgreen", "olive", "darkkhaki", "olivedrab", "lawngreen", "chartreuse", "darkgreen",
              "green", "forestgreen", "lime", "limegreen", "lightgreen", "palegreen", "springgreen", "seagreen", "mediumseagreen"]
    bblue = ["mediumaquamarine", "lightseagreen", "teal", "darkcyan", "midnightblue", "aqua", "cyan", "darkturquoise", "turquoise", "mediumturquoise", "aquamarine", "cadetblue",
             "steelblue", "dodgerblue", "deepskyblue", "cornflowerblue", "skyblue", "lightskyblue", "navy", "darkblue", "mediumblue", "mediumblue", "royalblue", ]
    orange = ["orange", "darkorange", "orangered", "coral",
              "tomato", "chocolate", "saddle brown", "sienna"]
    gray = ["dimgray", "gray",  "darkolivegreen", "darkgray", "silver", "lightgray", "gainsboro", "dimgrey",
            "grey", "darkgrey", "lightgrey", "lavender", "powderblue", "paleturquoise", "darkseagreen"]
    col = names[index]
    print("col", col)
    if col in rred:
        col = "red"
    elif col in yellow:
        col = "yellow"
    elif col in ggreen:
        col = "green"
    elif col in bblue:
        col = "blue"
    elif col in gray:
        col = "gray"
    elif col in orange:
        col = "orange"
    else:
        col = "not found"
    return col


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation


def openn(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection


def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def text(plate):
    cv2.imshow("down", plate)
    reader = easyocr.Reader(['ar'], gpu=False)
    result = reader.readtext(plate)
    result = result[0][1]
    # result = pytesseract.image_to_string(
    #     plate, lang="train", config=".")
    # print("plate number", result)
    # gray = get_grayscale(plate)
    # n = remove_noise(gray)
    # thresh = thresholding(n)
    # dil = dilate(thresh)
    # erodee = erode(dil)
    # opening = openn(erodee)
    # cannyy = canny(opening)
    # des = deskew(cannyy)
    # cv2.imshow("enhance", des)
    # result = pytesseract.image_to_string(plate, lang="train", config=".")
    return result


def display(im_path):
    dpi = 80
    # im_path = plt.imread(im_path)

    height, width = im_path.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_path, cmap='gray')

    plt.show()


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


# leftnew = remove_borders(left)
# rightnew = remove_borders(right)
# display(leftnew)

# display(right)


cropped = detection("./imgs/5.jpeg")
colorpart, textpart = cropping(cropped)
cv2.imshow("textxpart", textpart)
color = color_extraction(colorpart)
print("colorrr", color)
colortext = convert_rgb_to_names((color))
cv2.waitKey(0)
cv2.destroyAllWindows()
# charachters = text(textpart)
# color = colorextract(cropped_image
# print(f"color {colortext} and text {charachters}")
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# height, width = textpart.shape[:2]
# half_width = width // 2
# left = textpart[:, :half_width, :]
# right = textpart[:, half_width:, :]

# display(right)
# display(left)
img = cv2.cvtColor(textpart, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
img = cv2.medianBlur(img, 3)  # Median blur to remove noise
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img = cv2.medianBlur(img, 3)
img = noise_removal(img)
img = remove_borders(img)
display(img)
# imgnum = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
# imgnum = cv2.medianBlur(imgnum, 3)  # Median blur to remove noise
# imgnum = cv2.threshold(imgnum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# imgnum = cv2.medianBlur(imgnum, 3)
# imgnum = noise_removal(imgnum)
# imgnum = remove_borders(imgnum)
# cv2.imwrite("reight.jpg", img)
# cv2.imwrite("left.jpg", imgnum)


display(img)
# display(imgnum)


config = ('-l arabest --oem 3 --psm 11')
text1 = pytesseract.image_to_string(img, config=config)
config2 = ('-l  ara_number --oem 3 --psm 11')

# text2 = pytesseract.image_to_string(imgnum, config=config2)

print("chars", text1)
# print("digits", text2)
