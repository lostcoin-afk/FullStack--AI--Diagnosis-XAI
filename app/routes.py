from flask import Blueprint, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app.db import get_users_collection, get_visualisations_collection
from .model import run_inference  # You’ll define this next
from .utils import pil_image_to_base64
from PIL import Image

routes = Blueprint("routes", __name__)

@routes.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        image_file = request.files["image"]

        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join("app/static/uploads", filename)
            # image_file.save(filepath)

            pil_img = Image.open(filepath)
            encoded_img = pil_image_to_base64(pil_img)

            users_col = get_users_collection()
            users_col.insert_one({
                "patient_name": name,
                "phone_number": phone,
                "image_data": encoded_img
            })

            return redirect(url_for("routes.view_diagnosis", phone=phone))

    return render_template("upload.html")


@routes.route("/view_diagnosis")
def view_diagnosis():
    phone = request.args.get("phone")
    users_col = get_users_collection()
    user = users_col.find_one({"phone_number": phone})

    if not user:
        return "User not found", 404

    image_data = user["image_data"]  # base64 string

    # Run AI model + generate visualization
    diagnosis, processed_img = run_inference(image_data)

    # Save to visualisations collection
    visualisations_col = get_visualisations_collection()
    visualisations_col.insert_one({
        "phone_number": phone,
        "diagnosis": diagnosis,
        "processed_image": processed_img
    })

    # Render result page
    return render_template("view_diagnosis.html", name=user["patient_name"],
                           diagnosis=diagnosis, image_data=processed_img)
