import os
import sys
import time
import cv2
import numpy as np

import helper

directory = os.path.dirname(__file__)
sys.path.insert(0, directory)


def predict(video_path, out_video_path, checkpoint_path, device='cuda', show_bboxes=True):
    """
        Predict amount of persons in queue.

        video_path (str): path to input video
        out_video_path (str): path to output video
        checkpoint_path (str): path to checkpoint
        device (str): 'cuda' or 'cpu'. Default 'cuda'
        show_bboxes (bool): show bounding boxes or not. Default False
    """
    s_time = time.time()
    print('Starting inference...')

    video_path, _ = helper.smart_preprocess_video(video_path)
    out_video_root, _ = helper.split_to_dir_and_file(out_video_path)

    model = helper.init_yolo5(checkpoint_path, device)
    print(f'Detection model loaded at {round(time.time() - s_time, 2)}c')

    cap = cv2.VideoCapture(video_path)
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(out_video_root, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # mp4v VP80
    videoWriter = cv2.VideoWriter(out_video_path, fourcc, fps, (weight, height))
    print(f"Save video to {out_video_path}")

    step = 0
    mean_count = 0
    current_count = []
    queue_count = 0
    while cap.isOpened():
        print(f'Step {step}') if step % 20 == 0 else None
        if step % 21 == 0 or step == 0:
            st_t = time.time()

        flag, img = cap.read()
        if not flag:
            break

        results = model([img])
        current_count.append(len(results.pred[0]))
        if step == 0 or step % 10 == 0:
            mean_count = round(np.mean(current_count))
            current_count = []
        queue_count = mean_count if step == 0 or step % 20 == 0 else queue_count

        pred_image = np.array(results.get(show_bboxes)).reshape(height, weight, 3)
        queue_count_str = "  "  + str(queue_count) if queue_count < 10 else str(queue_count)
        pred_image = helper.add_text_to_image_pil(pred_image, f'Длина очереди: {queue_count_str}')
        print(f'Detected at {round(time.time()-st_t,2)}, queue count {queue_count}') if step % 20 == 0 else None

        videoWriter.write(pred_image)
        step += 1

    cap.release()
    videoWriter.release()
    os.remove(video_path)
    print(f'Processing time: {round(time.time() - s_time, 2)}c')


