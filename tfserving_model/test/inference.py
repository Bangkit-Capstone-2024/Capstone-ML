import requests
import json
import numpy as np
import cv2 as cv
import tensorflow as tf
import time


index_to_class = {
    "0": "baby_bed",
    "1": "baby_car_seat",
    "2": "baby_folding_fence",
    "3": "bathtub",
    "4": "booster_seats",
    "5": "bouncer",
    "6": "breast_pump",
    "7": "carrier",
    "8": "earmuffs",
    "9": "ride_ons",
    "10": "rocking_horse",
    "11": "sterilizer",
    "12": "stroller",
    "13": "walkers"
}

image = cv.imread("stroller2.png")

image = cv.resize(image, (224, 224))
image = image/255
image = np.expand_dims(image, axis=0)

start_time = time.time()
url = "http://localhost:8501/v1/models/image_model:predict"
data = json.dumps({"signature_name": "serving_default",
                  "instances": image.tolist()})
headers = {"content-type": "application/json"}
response = requests.post(url, data=data, headers=headers)
prediction = json.loads(response.text)["predictions"]
decoded_predictions = [
    index_to_class[str(np.argmax(pred))] for pred in prediction]
print()
print(prediction)
print()
print(decoded_predictions)
print()
print("Time taken : ", time.time() - start_time)
