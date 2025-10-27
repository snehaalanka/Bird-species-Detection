# Bird Species Detection using Deep Learning

This project is a **deep learning-based image classification model** that identifies bird species from input images using **transfer learning with MobileNetV2**.  
It leverages a **Convolutional Neural Network (CNN)** pre-trained on ImageNet to accurately predict bird species even with a limited dataset.

---

## 📋 Features
- Built using **TensorFlow** and **Keras**
- Utilizes **MobileNetV2** for transfer learning
- Supports **custom image predictions**
- Interactive **Streamlit web app** for real-time testing
- Well-structured dataset with training, testing, and unseen prediction folders

---

## Prerequisites
Make sure you have the following installed:
- Python 3.x  
- pip  
- Jupyter Notebook  
- Git  

---

## Getting Started

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/snehaalanka/Bird-species-Detection
cd Bird-species-Detection
```

### 2️⃣ Create a virtual environment
```bash
python3 -m venv venv
```

### 3️⃣ Activate the environment
**For Linux/macOS:**
```bash
source venv/bin/activate
```
**For Windows (PowerShell):**
```bash
venv\Scripts\activate
```

### 4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Dataset Structure
Your project folder should look like this:
```
Bird-species-Detection/
│
├── train/               # Training images (subfolders by species)
├── test/                # Testing images (subfolders by species)
├── images_to_predict/   # Images for custom prediction
├── bird_predictor_app.py
├── model/               # (Optional) Saved model files
├── requirements.txt
└── venv/
```

---

## 🧩 Running the Model (Jupyter Notebook)
To train or retrain the model:
```bash
jupyter notebook
```
Open your notebook (if available) and run all cells to train and save the model.

---

## 🌐 Running the Streamlit Web App
Once your environment is ready and the model is available, start the web app:
```bash
streamlit run bird_predictor_app.py
```

This will open the app in your default browser, allowing you to upload bird images and see predictions instantly.

---

## 📦 Dependencies
The main libraries used are:
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `streamlit`
- `opencv-python`
- `jupyter`

---

## 🧾 Notes
- Ensure your dataset folders are correctly named and structured.
- If you retrain the model, save it in a `/model` or `/saved_model` directory and update the path in `bird_predictor_app.py`.
- Streamlit and TensorFlow must be installed in the same Python environment for the app to work properly.

---

