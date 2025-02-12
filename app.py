# Barış Köse -2212721022

from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os


app = Flask(__name__)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "Resim dosyası gönderilmedi!", 400

    file = request.files["image"]
    process_type = request.form.get("type", "edges")

    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)


    if process_type == "edges":
        processed_img = cv2.Canny(img, 100, 200)
    elif process_type == "blur":
        processed_img = cv2.GaussianBlur(img, (15, 15), 0)
    elif process_type == "grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif process_type == "invert":
        processed_img = cv2.bitwise_not(img)
    elif process_type == "corner_harris":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        img[dst > 0.01 * dst.max()] = [0, 0, 255] 
        processed_img = img
    elif process_type == "sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        processed_img = cv2.sqrt(sobel_x**2 + sobel_y**2)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif process_type == "laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Laplacian(gray, cv2.CV_64F)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif process_type == "prewitt":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        grad_x = cv2.filter2D(gray, -1, prewitt_x)
        grad_y = cv2.filter2D(gray, -1, prewitt_y)
        processed_img = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    elif process_type == "roberts":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        grad_x = cv2.filter2D(gray, -1, roberts_x)
        grad_y = cv2.filter2D(gray, -1, roberts_y)
        processed_img = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    elif process_type == "erosion":
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.erode(img, kernel, iterations=1)
    elif process_type == "shi_tomasi":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        processed_img = img
    else:
        return "Geçersiz işlem türü!", 400


    output_dir = "static"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "output.jpg")
    cv2.imwrite(output_path, processed_img if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR))


    return send_from_directory(output_dir, "output.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
