from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import load_model, predict_with_cam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['CAM_FOLDER'] = 'static/cams'

# Class names per model
class_names_dict = {
    "leaf": ['Bacterial Blight', 'Cercospora Leaf Blight', 'Downey Mildew', 'Frogeye', 'Healthy', 'Potassium Deficiency', 'Rust', 'Target Spot'],

    "seed": ['Broken', 'Immature', 'Intact', 'Healthy', 'Skin Damaged', 'Spotted'],

}

# Load models
models = {
    "leaf": load_model("model/ASDID_maxvit_model.pth", len(class_names_dict["leaf"])),
    "seed": load_model("model/Soybean_Seeds_maxvit_model.pth", len(class_names_dict["seed"])),

}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    if model_name not in models:
        return "Model not found", 404

    prediction = None
    probabilities = []
    uploaded_image = ""
    cam_url = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cam_path = os.path.join(app.config['CAM_FOLDER'], filename)
            file.save(image_path)

            prediction, probabilities = predict_with_cam(
                models[model_name], image_path, cam_path, class_names_dict[model_name]
            )

            uploaded_image = filename
            cam_url = f"/static/cams/{filename}"

    return render_template(
        f"{model_name}.html",
        prediction=prediction,
        probabilities=probabilities,
        uploaded_image=uploaded_image,
        cam_url=cam_url,
        combined=zip(class_names_dict[model_name], probabilities)
    )

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CAM_FOLDER'], exist_ok=True)
    app.run(debug=True)
