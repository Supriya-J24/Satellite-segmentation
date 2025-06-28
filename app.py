from flask import Flask, request, jsonify, render_template
import numpy as np
import rasterio
import os
from imageio.v2 import imwrite  # Ensure correct import for writing images
from tensorflow import keras
from rasterio.enums import Resampling

app = Flask(__name__)

MODEL_PATH = r"models-final/unet_best_optical24_tf.keras"
segmentation_model = keras.models.load_model(MODEL_PATH)

OUTPUT_FOLDER = "static"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(file_path):
    """Load TIFF image using rasterio, extract RGB bands, and resize."""
    with rasterio.open(file_path) as dataset:
        num_bands = dataset.count
        if num_bands < 3:
            raise ValueError(f"Expected at least 3 bands, but found {num_bands}.")

        # Read only the first 3 bands (Red, Green, Blue)
        image = dataset.read([1, 2, 3], out_shape=(3, 256, 256), resampling=Resampling.bilinear)

        # Convert (bands, height, width) → (height, width, bands)
        image = np.transpose(image, (1, 2, 0))

        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0  

        # Add batch dimension for model input (1, 256, 256, 3)
        image = np.expand_dims(image, axis=0)

    return image

def convert_tiff_to_jpg(tiff_path, jpg_path):
    """Convert a TIFF image to JPG for visualization."""
    with rasterio.open(tiff_path) as dataset:
        image = dataset.read([1, 2, 3])  # Read RGB bands

        # Convert (Bands, H, W) → (H, W, Bands)
        image = np.transpose(image, (1, 2, 0))

        # Normalize to 8-bit (0-255) scaling
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Save as JPG
        imwrite(jpg_path, image)

    return jpg_path

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/version')
def version_info():
    import flask
    import tensorflow as tf
    import numpy as np
    import rasterio
    import imageio

    return jsonify({
        "Flask": flask.__version__,
        "TensorFlow": tf.__version__,
        "NumPy": np.__version__,
        "Rasterio": rasterio.__version__,
        "ImageIO": imageio.__version__
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        tiff_path = os.path.join(OUTPUT_FOLDER, "uploaded_image.tif")
        file.save(tiff_path)

        # Convert TIFF to JPG for display
        jpg_path = os.path.join(OUTPUT_FOLDER, "uploaded_image.jpg")
        convert_tiff_to_jpg(tiff_path, jpg_path)

        # Preprocess image for segmentation
        image = preprocess_image(tiff_path)

        prediction = segmentation_model.predict(image)
        predicted_mask = np.argmax(prediction[0], axis=-1)  # Convert to class labels

        # Convert mask into a color image
        segmented_image = np.zeros((256, 256, 3), dtype=np.uint8)
        colors = {
            0: [0, 100, 0],        # Tree Cover - Dark Green
            1: [255, 187, 34],     # Shrubland - Orange
            2: [255, 255, 76],     # Grassland - Yellow
            3: [240, 150, 255],    # Cropland - Pink
            4: [250, 0, 0],        # Built-up - Red
            5: [180, 180, 180],    # Bare/Sparse Vegetation - Gray
            6: [255, 255, 255],    # (Unused or Unclassified) - White
            7: [0, 100, 200],      # Permanent Water Bodies - Dark Blue
            8: [139, 69, 19],      # Bare Land - Brown
            9: [0, 255, 0],        # Vegetation - Bright Green
            10: [220, 220, 220]    # (Other category) - Light Gray
        }

        for class_idx, color in colors.items():
            segmented_image[predicted_mask == class_idx] = color

        # Save segmented output
        segmented_path = os.path.join(OUTPUT_FOLDER, "segmented_image.jpg")
        imwrite(segmented_path, segmented_image)

        return jsonify({
            "input_image_url": "/static/uploaded_image.jpg",
            "message": "Segmentation successful!",
            "segmented_image_url": "/static/segmented_image.jpg"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
