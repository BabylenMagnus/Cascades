import cv2


def video_iterator(video):

    while True:
        cap, frame = video.read()
        if not cap:
            break

        yield frame
