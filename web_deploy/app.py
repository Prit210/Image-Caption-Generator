from flask import Flask, render_template, request, url_for
import os
from inference import generate_caption

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            caption = generate_caption(filepath)
            image_url = url_for("static", filename=f"uploads/{file.filename}")

    return render_template("index.html", caption=caption, image_path=image_url)

if __name__ == "__main__":
    app.run(debug=True)

