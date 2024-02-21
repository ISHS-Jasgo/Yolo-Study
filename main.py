from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO
from ohshit import find_nearest
from test import test_case

import openvino as ov

core = ov.Core()
classification_model_xml = "model/yolov8n.xml"

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print(input_layer.shape)
print(output_layer.shape)

with open("model/metadata.yaml") as info:
    info_dict = yaml.load(info, Loader=yaml.Loader)

labels = info_dict["names"]
print(labels)


# Load the YOLOv8 model
def main(index):
    model = YOLO('yolov8n.pt')
    model.export(format='openvino')
    plt.figure(dpi=300)
    # Open the video file
    video_path = "http://cctvsec.ktict.co.kr/40/fNDqeWRt6jRsISwNhQ8gkIH5xCVlVBKZgBSn1xFuqEwSXPRcEQTQJMr5j4iMZ7bZ/w5wo7/emSS0JkfFRfuaew=="
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append([float(x), float(y)])

        else:
            print(find_nearest(list(track_history.values()), test_case) / 30 * 100)
            plt.savefig(f"test{index}.png")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(5)
