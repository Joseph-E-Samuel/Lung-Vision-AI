from flask import Flask, request, json, jsonify
from cancer_model_init import model_init
import tensorflow as tf
from uuid import uuid4
import os
import cv2
import numpy as np

api = Flask(__name__)


@api.route("/upload/", methods=["POST"])
def post_orders(image):
    response = {}

    if request.is_json:
        payload = request.get_json()
        if not os.path.isfile("cancer_detection_model.keras"):
            model_init()

        model = tf.keras.models.load_model("cancer_detection_model.keras")

        print(payload)

        ref_id = payload.get("ref_id")
        # 6648c703-9914-41a0-b8e9-11ea95ae5d90
        image_id = str(uuid4())

        image = payload.get("image")

        img = cv2.imread("LungCancerCTscan.jpg")
        img = cv2.resize(img, (430, 305))
        img = np.array(img)
        model = tf.keras.models.load_model("cancer_detection_model.keras")

        predicted_condition = model.predict(img)

        response["id"] = image_id
        response["image"] = image
        response["predicted_condition"] = predicted_condition


        return predicted_condition[0]
    return 500


if __name__ == "__main__":
    api.run()
