import argparse
from module import predict

"""
    Test function to run from colab or cmd.
    Example: 'python test.py stock1.mp4 stock.webm yolov5l.py --show-bboxes'
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_video_path', type=str, help='path to the input video')
    parser.add_argument('out_video_path', type=str, help='path to the output video')
    parser.add_argument('detection_weights', type=str, help='detection checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='device to predict on')
    parser.add_argument('--hide-bboxes', action='store_false', help='show bounding boxes or not')

    args = parser.parse_args()

    output_path = predict(args.in_video_path, args.out_video_path, args.detection_weights, args.device, args.hide_bboxes)

