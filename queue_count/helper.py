import os
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolov5 import hubconf


def init_yolo5(ckpt_path, device='cuda'):
    model = hubconf.create_new(name='yolov5l', pretrained=True, channels=3, classes=80, autoshape=True,
                               ckpt_path=ckpt_path, device=device)
    model.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model.classes = [0]

    return model.to(device)


def smart_preprocess_video(path):
    """
        The function analise input video file and make smart decision to reduce video or not.
        Video may be reduced depends on size of file, amount of frames or/and resolution.
        New video will be saved in the same folder as original and will be removed after prediction.
    """

    MAX_RESOLUTION = 640
    MAX_SIZE = 10 * 1024 * 1024
    MAX_FRAMES = 400

    s_time = time.time()
    folder, fname = split_to_dir_and_file(path)

    cap = cv2.VideoCapture(path)
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = os.path.getsize(path)
    print(f'Initial video resolution: {weight}/{height}, frames: {frames}, size {size}')

    if size < MAX_SIZE / 2 and frames < MAX_FRAMES / 2:
        scale, frame_reduce = 1, 1
    elif size < MAX_SIZE and frames < MAX_FRAMES and max(weight, height) <= MAX_RESOLUTION:
        scale, frame_reduce = 1, 0.5
    else:
        scale, frame_reduce = round(MAX_RESOLUTION / max(weight, height), 2), 0.5

    w, h = int(weight * scale), int(height * scale)
    fps = int(cap.get(cv2.CAP_PROP_FPS) * frame_reduce)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(folder, 'reduced_' + str(scale) + '_' + str(frame_reduce) + '_' + fname)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    step = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(
                f'Video was reduced to {w}/{h}, got every {frame_reduce} frame, time: {np.round(time.time() - s_time, 2)}')
            break
        if step % (1 / frame_reduce) == 0:
            frame = cv2.resize(frame, (w, h))
            out.write(frame)
        step += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path, fname


def split_to_dir_and_file(path):
    idx = path.rfind('/')
    return path[:idx + 1], path[idx + 1:]


def add_text_to_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 40)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2  # Line thickness of 2 px
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def add_text_to_image_pil(image, text):
  '''
  Add text to image. Work with utf-8 chars
    image - np.array image
    text - text to add
  return image as np.array
  '''
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font_size = max(round(max(img.size) / 20), 18) # dynamic font size
    font = ImageFont.truetype(str(os.path.dirname(os.path.abspath(__file__))) + "/resources/arial.ttf", font_size)

    text_width, text_height = font.getsize(text) # get text size in current font
    img_width, img_height = img.size;
    rec_xy = [(0, img_height-10-text_height),(img_width, img_height-8)]
    # (124, 158, 0) - зеленый, (242, 115, 83)(245, 129, 113) - всетло красный (197, 122,0) - синий (91, 164, 222) - коричневый
    draw.rectangle(rec_xy, fill=(222, 164, 91), outline=None, width=1) 

    draw.text((img_width/2 - text_width/2, img_height-8-text_height), text, fill=(255, 255, 255), font=font) 
    return np.asarray(img)
