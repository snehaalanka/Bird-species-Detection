# Bird-species-Detection
This project is a deep learning-based image classification model that identifies bird species from input images using transfer learning with MobileNetV2. The dataset consists of labeled bird images categorized into folders for training, testing, and unseen prediction.
We utilized a Convolutional Neural Network (CNN) model pre-trained on ImageNet to benefit from prior learned features. The model was fine-tuned on our custom bird species dataset, enabling accurate predictions even with limited data.

PREREQUISITES:
Make sure you have the following installed:
-Python 3.x
-pip
-Jupyter Notebook
-Streamlit
-TensorFlow

Your project should contain a folder called Bird-species-Detection/ with:
-train/ folder (for training)
-test/ folder (for evaluation)
-images_to_predict/ folder (for testing on new images)
