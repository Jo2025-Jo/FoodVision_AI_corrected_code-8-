
import os
import json
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from torchvision import models, transforms
from werkzeug.utils import secure_filename


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FOODVISION_SECRET",
    "change-this-secret-before-public-deployment"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")

MODEL_PATH = os.path.join(MODEL_DIR, "foodvision_model.pth")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE


# --------------------------------------------------
# PYTORCH CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 160

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = None
class_names = []
model_error = None


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():

    global model, class_names, model_error

    try:

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                "Missing models/foodvision_model.pth"
            )

        if not os.path.isfile(CLASSES_PATH):
            raise FileNotFoundError(
                "Missing models/class_names.json"
            )

        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            names_data = json.load(file)

        if isinstance(names_data, list):

            class_names = [str(name) for name in names_data]

        elif isinstance(names_data, dict):

            class_names = [
                str(names_data[key])
                for key in sorted(names_data, key=lambda x: int(x))
            ]

        else:
            raise ValueError(
                "class_names.json must contain a list or dictionary."
            )

        if not class_names:
            raise ValueError("No food categories found.")

        network = models.mobilenet_v3_small(weights=None)

        input_features = network.classifier[3].in_features

        network.classifier[3] = nn.Linear(
            input_features,
            len(class_names)
        )

        try:
            saved_weights = torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=True
            )
        except TypeError:
            saved_weights = torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )

        if isinstance(saved_weights, dict):

            if "state_dict" in saved_weights:
                saved_weights = saved_weights["state_dict"]

            elif "model_state_dict" in saved_weights:
                saved_weights = saved_weights["model_state_dict"]

        network.load_state_dict(saved_weights)

        network.to(DEVICE)
        network.eval()

        model = network
        model_error = None

        print("FoodVision AI model loaded successfully.")
        print("Number of food classes:", len(class_names))
        print("Running device:", DEVICE)

    except Exception as error:

        model = None
        model_error = str(error)

        print("MODEL LOADING ERROR:", model_error)


load_model()


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# GENERATE BLUE GRAD-CAM HEATMAP
# --------------------------------------------------

def generate_gradcam(pil_image, target_class):

    if model is None:
        raise RuntimeError("AI model is not loaded.")

    captured = {}

    def capture_activation(module, inputs, output):

        output.retain_grad()
        captured["activation"] = output

    target_layer = model.features[-1]

    hook = target_layer.register_forward_hook(
        capture_activation
    )

    try:

        image_tensor = transform(
            pil_image.convert("RGB")
        ).unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            output = model(image_tensor)

            output[0, target_class].backward()

        activation = captured["activation"]
        gradient = activation.grad

        if gradient is None:
            raise RuntimeError("Grad-CAM gradients unavailable.")

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(dim=1).squeeze(0)

        cam = torch.relu(cam).detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        original = np.array(pil_image.convert("RGB"))

        height, width = original.shape[:2]

        cam = cv2.resize(cam, (width, height))

        heatmap = np.uint8(255 * cam)

        # Blue/cyan colour heatmap
        coloured = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_OCEAN
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            coloured,
            0.45,
            0
        )

        filename = uuid.uuid4().hex + "_heatmap.jpg"

        output_path = os.path.join(
            HEATMAP_DIR,
            filename
        )

        success = cv2.imwrite(output_path, overlay)

        if not success:
            raise RuntimeError("Could not save heatmap.")

        return filename

    finally:
        hook.remove()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        expected_username = os.environ.get(
            "FOODVISION_USERNAME", "admin"
        )

        expected_password = os.environ.get(
            "FOODVISION_PASSWORD", "foodvision123"
        )

        if (
            username == expected_username
            and password == expected_password
        ):

            session["logged_in"] = True
            session["username"] = username

            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


# --------------------------------------------------
# DASHBOARD AND FOOD IMAGE ANALYSIS
# --------------------------------------------------

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":

        if model is None:

            flash(
                "AI model is unavailable: " + str(model_error),
                "error"
            )

            return render_template(
                "dashboard.html",
                username=session.get("username", "User"),
                model_ready=False,
                model_error=model_error
            )

        uploaded_file = request.files.get("food_image")

        if not uploaded_file or not uploaded_file.filename:

            flash("Please select a food image.", "error")

            return redirect(url_for("dashboard"))

        if not allowed_file(uploaded_file.filename):

            flash("Unsupported image format.", "error")

            return redirect(url_for("dashboard"))

        try:

            image = Image.open(uploaded_file.stream).convert("RGB")

            # Save original image as JPEG
            image_filename = uuid.uuid4().hex + ".jpg"

            image_path = os.path.join(
                UPLOAD_DIR,
                image_filename
            )

            image.save(image_path, format="JPEG", quality=95)

            # --------------------------------------
            # MODEL PREDICTION
            # --------------------------------------

            image_tensor = transform(
                image
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[0]

                count = min(5, len(class_names))

                scores, indices = torch.topk(
                    probabilities,
                    k=count
                )

            predicted_index = int(indices[0].item())

            predicted_name = class_names[
                predicted_index
            ].replace("_", " ").title()

            confidence = round(
                float(scores[0].item()) * 100,
                2
            )

            top_results = []

            for score, index in zip(scores, indices):

                top_results.append({

                    "name": class_names[
                        int(index.item())
                    ].replace("_", " ").title(),

                    "confidence": round(
                        float(score.item()) * 100,
                        2
                    )

                })

            # --------------------------------------
            # IMAGE ANALYSIS USING OPENCV
            # --------------------------------------

            image_array = np.array(image)

            gray = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2GRAY
            )

            brightness = round(
                float(np.mean(gray)),
                2
            )

            sharpness = round(
                float(
                    cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()
                ),
                2
            )

            edges = cv2.Canny(
                gray,
                100,
                200
            )

            edge_ratio = round(
                float(np.count_nonzero(edges))
                / edges.size * 100,
                2
            )

            # --------------------------------------
            # GRAD-CAM HEATMAP
            # --------------------------------------

            heatmap_filename = generate_gradcam(
                image,
                predicted_index
            )

            return render_template(

                "result.html",

                username=session.get("username", "User"),

                image_filename=image_filename,

                heatmap_filename=heatmap_filename,

                predicted_name=predicted_name,

                confidence=confidence,

                top_results=top_results,

                brightness=brightness,

                sharpness=sharpness,

                edge_ratio=edge_ratio

            )

        except Exception as error:

            app.logger.exception(
                "Food image analysis failed"
            )

            flash(
                "Image analysis failed: " + str(error),
                "error"
            )

            return redirect(url_for("dashboard"))

    return render_template(

        "dashboard.html",

        username=session.get("username", "User"),

        model_ready=model is not None,

        model_error=model_error

    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# --------------------------------------------------
# HEALTH CHECK FOR HOSTING
# --------------------------------------------------

@app.route("/health")
def health():

    return {

        "status": "ok",

        "model_ready": model is not None,

        "class_count": len(class_names),

        "device": str(DEVICE),

        "model_error": model_error

    }


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    flash("Image is too large. Maximum size is 10 MB.", "error")

    return redirect(url_for("dashboard"))


# --------------------------------------------------
# RUN LOCAL OR HOSTED SERVER
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
