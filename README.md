# Image Recognition API (ResNet18 + FastAPI + Docker)

This project is an end-to-end image recognition system built using a pre-trained deep learning model. The goal is to classify input images and return the **Top 5 most likely predictions**, exposing the model through a FastAPI REST API and deploying it using Docker.

---

## 1. Model Selection & Setup

In this project, we use a **pre-trained ResNet18 model** from PyTorch’s `torchvision` library. The model has been trained on the ImageNet dataset, which contains millions of labeled images across 1000 classes.

The model is loaded in evaluation mode and used directly for inference, without additional training.

We also load the ImageNet class labels from a JSON file to map model outputs to human-readable categories.

---

## 2. Image Preprocessing

Before feeding an image into the model, we apply a preprocessing pipeline to ensure compatibility with ResNet18 requirements:

- Resize image to **224x224**
- Convert image to RGB format
- Transform image into a tensor
- Normalize input format for PyTorch model

This ensures consistent input structure for accurate predictions.

---

## 3. Inference Pipeline (Top 5 Predictions)

Once the image is processed, it is passed through the model to generate predictions.

We apply a **Softmax function** to convert raw logits into probabilities, and then extract the **Top 5 most likely classes** using `torch.topk`.

Each prediction includes:
- Label (class name)
- Confidence score

This allows users to understand not only the predicted class but also the model’s certainty.

---

## 4. FastAPI Development

The trained model is wrapped inside a **FastAPI application** to expose it as a REST API.

The API includes:

- `GET /` → Health check endpoint
- `POST /predict` → Accepts image upload and returns Top 5 predictions

FastAPI also provides an interactive documentation interface via Swagger UI, allowing easy testing of the model.

---

## 5. Local Testing

Before containerization, the system was tested locally to ensure correct functionality:

- Verified model loading and inference in Python
- Tested preprocessing pipeline with sample images
- Confirmed correct Top 5 predictions output
- Validated FastAPI endpoints

This ensured the system was working correctly before deployment.

---

## 6. Docker Containerization

To ensure reproducibility and portability, the application was containerized using Docker.

A Docker image was created including:
- Python environment
- Required dependencies
- Model code and API

---

## 7. Deployment Verification

After containerization, the system was validated by:

- Running the Docker container successfully
- Accessing the API
- Testing image uploads through Swagger UI
- Confirming identical predictions to local execution

This ensured that the Dockerized version behaves exactly like the local version.

---

## 8. Key Features

- Pre-trained deep learning model (ResNet18)
- Image classification using ImageNet labels
- Top 5 prediction output with confidence scores
- REST API built with FastAPI
- Dockerized deployment for reproducibility
- End-to-end ML inference pipeline

---

## 9. API Testing Evidence

A screenshot demonstrating a successful image recognition inference has been included in the `screenshots` folder. This captures a real API request executed through the FastAPI endpoint, validating the correct functioning of the deployed model and confirming consistent inference results in a production-like environment.


## 10. Future Improvements

- Add GPU (CUDA) support for faster inference
- Replace ResNet18 with more advanced architectures
- Add batch prediction endpoint
- Deploy to cloud (AWS / Azure / Hugging Face Spaces)
- Add authentication layer for API security
- Build a simple frontend for image upload and visualization
