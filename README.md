# AI vs Real Image Detection

An AI-based image classification system that detects whether an uploaded image is **Real** or **AI-Generated**.

## 🚀 Project Overview

This project uses a deep learning model to classify images into two categories:

- **Real Image**
- **AI-Generated Image**

The model is trained using the **OpenFake dataset** and uses **ResNet18** as the image classification architecture.

The trained model is integrated with **Django** to provide a simple web interface where users can upload an image and receive a prediction along with a confidence score.

## ✨ Features

- Upload an image through a web interface
- Detect whether the image is Real or AI-Generated
- Display prediction confidence
- Image preprocessing and normalization
- Data augmentation during training
- ResNet18-based image classification
- Django web integration
- Responsive and modern UI

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Django
- Pillow (PIL)
- OpenFake Dataset
- HTML
- CSS
- JavaScript

## 🧠 Model

The project uses **ResNet18**, a convolutional neural network architecture designed for image classification.

The images are resized to **224 × 224 pixels** and normalized using ImageNet mean and standard deviation values.

### Training Augmentation

During training, the following augmentations are applied:

- Random Resized Crop
- Random Horizontal Flip
- Random Rotation
- Color Jitter

## 📊 Dataset

The project uses the **OpenFake dataset** containing both real and AI-generated images.

For this project, a balanced subset of:

- 500 Real images
- 500 AI-generated images

was used.

The dataset is divided into training, validation, and test sets.

> Dataset images are not included in this repository because of their size.

## 🌐 Django Integration

The trained ResNet18 model is integrated into a Django application.

Users can:

1. Open the web application.
2. Upload an image.
3. The image is preprocessed.
4. The trained model performs inference.
5. The application displays the predicted class and confidence score.

## 📁 Project Structure

```text
AI-VS-REAL-IMAGE-DETECTION/
│
├── dataset/                    # Dataset (not uploaded)
├── dataset_split/              # Train/validation/test data (not uploaded)
│
├── preprocessing.py            # Image preprocessing & augmentation
├── download_dataset.py         # Download dataset
├── split_dataset.py            # Split dataset
├── train.py                    # Model training
├── evaluate.py                 # Model evaluation
├── predict.py                  # Image prediction
├── check_dataset.py            # Dataset checking
├── check_labels.py             # Label checking
├── inspect_samples.py          # Inspect dataset samples
├── Verify_preprocessing.py     # Verify preprocessing
│
├── webapp/
│   ├── manage.py               # Django management file
│   │
│   ├── detector/
│   │   ├── migrations/
│   │   ├── templates/
│   │   │   └── detector/
│   │   │       └── upload.html # Web UI
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── urls.py
│   │   ├── views.py            # Prediction logic
│   │   └── tests.py
│   │
│   └── webapp/
│       ├── __init__.py
│       ├── asgi.py
│       ├── settings.py
│       ├── urls.py
│       └── wsgi.py
│
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NisuBharti32/AI-VS-REAL-IMAGE-DETECTION.git
cd AI-VS-REAL-IMAGE-DETECTION
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install torch torchvision pillow django datasets
```

### 5. Run the Django Application

Navigate to the Django project:

```bash
cd webapp
```

Start the development server:

```bash
python manage.py runserver
```

### 6. Open the Application

Open the following URL in your browser:

```text
http://127.0.0.1:8000/
```

The web application is now ready to upload an image and get a prediction.

## ▶️ Running the Project

### 1. Activate the Virtual Environment

For Windows:

```powershell
.venv\Scripts\activate
```

### 2. Navigate to the Django Application

```bash
cd webapp
```

### 3. Start the Django Development Server

```bash
python manage.py runserver
```

### 4. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

### 5. Upload an Image

- Click on the **Upload Image** area.
- Select a JPG, JPEG, or PNG image.
- Click the **Predict** button.
- The model will classify the image as **Real** or **AI-Generated**.
- The prediction confidence will also be displayed.

### 6. Command-Line Prediction

The trained model can also be used without the Django web interface:

```bash
python predict.py "path/to/image.jpg"
```

Example:

```bash
python predict.py "C:\Users\YourName\Downloads\test.jpg"
```

## 🔮 Future Improvements

- Train the model on a larger and more diverse dataset.
- Improve detection accuracy for different types of AI-generated images.
- Improve performance on group photographs.
- Experiment with advanced CNN and Vision Transformer architectures.
- Add Grad-CAM for model explainability.
- Deploy the application online.

## ⚠️ Limitations

The model is trained on a relatively small dataset subset, so predictions may not always be accurate.

The model can be affected by factors such as:

- Image quality
- Image composition
- Number of people in the image
- Image generation method
- Differences between training and real-world images

Therefore, the prediction should be considered a **model estimate and not a guaranteed determination**.

## 👩‍💻 Author

**Nisu Bharti**

GitHub: [NisuBharti32](https://github.com/NisuBharti32)
