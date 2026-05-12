from flask import Flask, send_file
from PIL import Image, ImageDraw
import io

app = Flask(__name__)

@app.route("/image")
def get_image():

    # Create sample image
    img = Image.new("RGB", (400, 300), color="white")

    draw = ImageDraw.Draw(img)
    draw.text((120, 140), "Hello Android This is an Image sample", fill="black")

    # Convert image to memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Return image as response
    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)