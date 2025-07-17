

---

# Human Stress Detection in and through Sleep
## 🌐 Deployment - 🚀 Live Frontend (Streamlit)

Access the deployed frontend here:  

[![Open in Streamlit](https://img.shields.io/badge/Launch%20App-Streamlit-orange?style=for-the-badge&logo=streamlit)](https://stress-detector-a23.streamlit.app/)

<img width="964" height="899" alt="image" src="https://github.com/user-attachments/assets/c28e9754-4fcc-48c1-919a-9fd5f6f235d3" />


## 🛠️ Backend API (Flask)
The Flask API is deployed on Railway and can be accessed here:

```bash
https://biology-backend-production.up.railway.app/predict
```
You can send a POST request to this endpoint with the required JSON payload to get stress predictions.



## 🧠 Project Overview

This project detects **stress levels** in humans during sleep using **physiological data** collected from **IoT-based devices**. It leverages a machine learning model trained on a dataset with features like **snoring rate**, **body temperature**, **sleeping hours**, and **heart rate**.

The data comes from **Kaggle** and the **SaYoPillow** project, an IoT-based smart pillow that monitors sleep and helps detect stress. The aim is to assist users in understanding and managing stress for improved **health and well-being**.

---

## 🧩 Problem Description

The project predicts **stress levels** based on sleep cycle data. Stress is detected using features like **heart rate**, **snoring rate**, and **body temperature**, and is categorized into **five levels** ranging from **low to high**.

The system aims to improve sleep quality and offer personalized stress tracking insights, helping users manage their well-being effectively.

---

## 📊 Dataset

The dataset used for training the model includes the following features:

| Feature Name         | Description                                     |
| -------------------- | ----------------------------------------------- |
| **Snoring Rate**     | Intensity of snoring during sleep (0-100 scale) |
| **Body Temperature** | Body temperature while sleeping (°C)            |
| **Sleeping Hours**   | Total number of hours of sleep                  |
| **Heart Rate**       | Heart rate during sleep (beats per minute)      |
| **Stress Level**     | Detected stress level during sleep (1-5 scale)  |

### Stress Level Categories:

* **1** – Low (No Stress)
* **2** – Medium-Low (Mild Stress)
* **3** – Medium (Moderate Stress)
* **4** – Medium-High (Significant Stress)
* **5** – High (Concerning Stress)

---

## 🧑‍💻 Model Training

### Data Preprocessing

The dataset is read from a CSV file (`stress_data.csv`) with the following columns:

* **snoring\_rate**
* **respiration\_rate**
* **body\_temperature**
* **sleeping\_hours**
* **heart\_rate**
* **stress\_level**

After loading and cleaning the data, the features are selected to be used for training. The most important features used for predicting stress level are:

* **heart\_rate**
* **sleeping\_hours**
* **snoring\_rate**
* **body\_temperature**

### Model

The model is built using **Logistic Regression**. The training process involves:

* Splitting the data into training and validation sets.
* Preprocessing the features with **DictVectorizer** to convert them into a numerical format.
* Training the **Logistic Regression** model on the preprocessed data.

The final model is saved as `final_model.bin` which includes the trained vectorizer and model.

---

## 🔧 How to Run

### Install Dependencies

To run the project, make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt
```

### Starting the API Server

To start the Flask API server that serves predictions, run:

```bash
python app.py
```

This will start a server on `http://127.0.0.1:5000`.

### Streamlit Frontend

To launch the Streamlit frontend:

```bash
streamlit run frontend.py
```

This will open the Streamlit app in your browser, where you can input data and get stress predictions.

---

## 💡 Example Input

Here's an example of a **medium stress** input for testing:

```json
{
  "heart_rate": 85.0,
  "sleeping_hours": 6.5,
  "snoring_rate": 15.0,
  "body_temperature": 37.2
}
```

### Expected Output:

For medium stress levels, the response should look like this:

```json
{
  "stress_level": 3,
  "description": "Medium (Moderate Stress)"
}
```

---

## 🚀 Contributing

If you'd like to contribute to the project, feel free to fork the repository and make changes. Pull requests are always welcome!


