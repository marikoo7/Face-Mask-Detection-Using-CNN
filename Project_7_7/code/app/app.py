from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = "final_model.h5"
model = load_model(MODEL_PATH)


class_labels = ["With Mask", "Without Mask"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(128, 128)).convert('RGB')
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)


        preds = model.predict(x)
        print("Raw predictions:", preds)
        result = class_labels[int(np.round(preds[0][0]))]

        return render_template("result.html", result=result, img_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
