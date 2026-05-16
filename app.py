# =========================
# app.py
# =========================

from flask import Flask, render_template, request

import tensorflow as tf
from PIL import Image

import numpy as np

import base64
import io
import cv2
import os

from datetime import datetime

# =========================
# CREATE FLASK APP
# =========================

app = Flask(__name__)

# =========================
# CREATE FEEDBACK FOLDER
# =========================

if not os.path.exists("feedback"):

    os.makedirs("feedback")

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("model.h5")

print("MODEL LOADED SUCCESSFULLY")

# =========================
# GLOBAL IMAGE
# =========================

last_uploaded_image = None

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_image(img):

    # TO NUMPY
    img = np.array(img)

    # =========================
    # THRESHOLD
    # =========================

    _, img = cv2.threshold(
        img,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =========================
    # FIND CONTOURS
    # =========================

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =========================
    # GET BIGGEST CONTOUR
    # =========================

    if len(contours) > 0:

        c = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(c)

        img = img[y:y+h, x:x+w]

    # =========================
    # ADD PADDING
    # =========================

    img = cv2.copyMakeBorder(
        img,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # =========================
    # RESIZE
    # =========================

    img = cv2.resize(
        img,
        (28, 28)
    )

    # =========================
    # NORMALIZE
    # =========================

    img = img / 255.0

    # =========================
    # RESHAPE
    # =========================

    img = img.reshape(
        1,
        28,
        28,
        1
    )

    return img

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template("index.html")

# =========================
# PREDICT IMAGE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    global last_uploaded_image

    # GET FILE
    file = request.files["image"]

    # OPEN IMAGE
    img = Image.open(file).convert("L")

    # SAVE ORIGINAL IMAGE
    last_uploaded_image = img.copy()

    # =========================
    # IMAGE TO BASE64
    # =========================

    buffer = io.BytesIO()

    img.save(buffer, format="PNG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # =========================
    # PREPROCESS
    # =========================

    processed_img = preprocess_image(img)

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(processed_img)

    # RESULT
    result = int(
        np.argmax(prediction[0])
    )

    # CONFIDENCE
    confidence = float(
        np.max(prediction[0]) * 100
    )

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence, 2),
        uploaded_image=image_base64
    )

# =========================
# PREDICT CANVAS
# =========================

@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():

    global last_uploaded_image

    # GET JSON
    data = request.json["image"]

    # REMOVE HEADER
    data = data.split(",")[1]

    # DECODE IMAGE
    image_bytes = base64.b64decode(data)

    # OPEN IMAGE
    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    # SAVE ORIGINAL
    last_uploaded_image = img.copy()

    # PREPROCESS
    processed_img = preprocess_image(img)

    # PREDICT
    prediction = model.predict(processed_img)

    # RESULT
    result = int(
        np.argmax(prediction[0])
    )

    # CONFIDENCE
    confidence = float(
        np.max(prediction[0]) * 100
    )

    return {
        "result": result,
        "confidence": round(confidence, 2)
    }

# =========================
# SAVE FEEDBACK
# =========================

@app.route("/feedback", methods=["POST"])
def feedback():

    global last_uploaded_image

    correct_digit = request.form["correct_digit"]

    # CREATE FOLDER
    folder = f"feedback/{correct_digit}"

    if not os.path.exists(folder):

        os.makedirs(folder)

    # FILE NAME
    filename = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    ) + ".png"

    # FULL PATH
    path = os.path.join(
        folder,
        filename
    )

    # SAVE IMAGE
    if last_uploaded_image is not None:

        last_uploaded_image.save(path)

    return render_template(
        "index.html",
        message="Feedback saved successfully!"
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

      app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )