# Input-independent code (line numbers in this file):
#   model preparation: 33–40.
#   local setup and model loading: 78–86.

import os
import io
import json
import sys
from . import storage

import cv2

client = storage.storage.get_instance()

labels = ["person", "bicycle", "car", "motorcycle",
"airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
"snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
"spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
"pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
"toilet", "tv", "laptop", "mouse", "remote", "keyboard",
"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]


def load_model(bucket, weights_blob, config_blob, dest_dir):
    # ---- Input-independent BEGIN: model preparation ----
    # Model file locations can be obtained from workflow configuration before upstream functions complete.
    # Therefore, model preparation is input-independent and can execute without waiting for upstream output.
    weights_path = os.path.join(dest_dir, "model.weights")
    client.download(bucket, weights_blob, weights_path)

    config_path = os.path.join(dest_dir, "model.config")
    client.download(bucket, config_blob, config_path)

    net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
    return net
    # ---- Input-independent END: model preparation ----


def load_frames(benchmark_bucket, bucket, blobs, dest_dir):
    # ---- Input-dependent BEGIN: frame retrieval ----
    for blob in blobs:
        stripped_blob = blob.replace(bucket + '/', '')
        path = os.path.join(dest_dir, stripped_blob)
        client.download(benchmark_bucket, blob, path)
        yield cv2.imread(path)
    # ---- Input-dependent END: frame retrieval ----


def detect(net, img):
    # ---- Input-dependent BEGIN: image preprocessing and inference ----
    rows = img.shape[0]
    cols = img.shape[1]
    img = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
    net.setInput(img)
    out = net.forward()

    preds = []
    for detection in out[0,0,:,:]:
        score = float(detection[2])
        if score > 0.5:
            class_id = int(detection[1])
            preds.append({
                "class": labels[class_id],
                "score": score
            })

    return preds
    # ---- Input-dependent END: image preprocessing and inference ----


def handler(event):
    # ---- Input-independent BEGIN: local setup and model loading ----
    tmp_dir = "/tmp"

    # With DASH, these inputs can be bound as references before upstream values are available.
    # Therefore, these assignments are input-independent and can execute without waiting for upstream output.
    benchmark_bucket = event["benchmark_bucket"]

    # The bucket and model locations can be obtained from workflow configuration before upstream functions complete.
    # Therefore, model loading is input-independent and can execute without waiting for upstream output.
    net = load_model(benchmark_bucket, event["model_bucket"] + '/' + event["model_weights"], event["model_bucket"] + '/' + event["model_config"], tmp_dir)
    # ---- Input-independent END: local setup and model loading ----

    # ---- Input-dependent BEGIN: frame retrieval, inference, and result construction ----
    frames = list(load_frames(benchmark_bucket, event["frames_bucket"], event["frames"], tmp_dir))
    preds = [detect(net, frame) for frame in frames]

    frames_names = event["frames"]
    frames_names = [x.split(".")[0] for x in event["frames"]]

    preds = {f"{frames_names[idx]}": dets for idx, dets in enumerate(preds)}

    return preds
    # ---- Input-dependent END: frame retrieval, inference, and result construction ----
