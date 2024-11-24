# Photo Colorization
This project uses deep learning techniques to colorize black and white images. It employs a pre-trained model based on the Caffe framework to achieve this. The primary goal is to take a grayscale image and produce a colorized version of it.

## Requirements
  - To run this project, you will need the following libraries:

    - OpenCV
    - NumPy
    - PIL (Pillow)
  
  - You can install these libraries using pip:
``` bash
pip install opencv-python numpy Pillow
```

## File Structure
The project consists of the following files:
```
colorize_photo/
├── model/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   ├── pts_in_hull.npy
│   └── lion.jpg
└── colorizePhotos.py
```

  - colorization_deploy_v2.prototxt: The model architecture file.
  - colorization_release_v2.caffemodel: The pre-trained model weights.
  - pts_in_hull.npy: The kernel file for the model.
  - lion.jpg: An example black and white image for testing.
  - colorizePhotos.py: The main script to run for colorizing images.

## Usage
  1. Prepare Your Image: Place your black and white image in the model/ directory or update the image_path variable in colorizePhotos.py with the path to your image.
  2. Run the Script: Execute the script using Python:
```bash
python colorizePhotos.py
```

  3. View Results: The script will display the original black and white image and the colorized image. The colorized image will also be saved as colorized.png in the current directory.

## How It Works
  1. The script reads a black and white image and normalizes it.
  2. It converts the image from BGR to LAB color space.
  3. The lightness channel (L) is extracted and processed through a neural network model to predict the color channels (A and B).
  4. The predicted channels are resized to match the original image dimensions and combined with the lightness channel to form a colorized LAB image.
  5. Finally, the LAB image is converted back to BGR format and saved.
