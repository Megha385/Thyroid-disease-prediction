# ThyroCheck – Thyroid Disease Prediction & Health Assessment

This project is a complete end-to-end system for predicting thyroid disorders and generating personalized health assessment reports. The system analyzes patient lab values (TSH, T3, T4, TT4, age, gender, symptoms, etc.) and predicts whether the condition is **Normal**, **Hypothyroid**, or **Hyperthyroid**.  
It includes a backend service, HTML dashboard, PDF report generation, and data-driven health insights.

---

## ⭐ Features

### 🔍 Machine Learning Prediction
- Predicts **Normal / Hypothyroid / Hyperthyroid** using cleaned medical dataset  
- Uses algorithms like Random Forest, SVM, Logistic Regression  
- Includes a trained model loaded through `predict_model.py`

### 📊 Interactive Health Dashboard
- Frontend with HTML, CSS, and JS  
- Displays prediction results  
- Shows nutritional suggestions and exercise guidance  
- User-friendly interface for patients

### 🧾 PDF Medical Report Generation
- Generates professional thyroid health reports using ReportLab  
- Includes:
  - Patient details  
  - Prediction result  
  - Key medical indicators  
  - Recommended diet & lifestyle tips  

### 📧 Email Report Service (Optional)
- Sends personalized PDF reports to users via email  
- Integrated through Flask backend

### 🔐 Backend (Flask)
- APIs to process patient input  
- Prediction API for ML model  
- Secure data handling and validation  
- Supports communication with frontend (CORS enabled)

---

## 📂 Project Structure

## 🚀 How to Run This Project

### 1️⃣ Install dependencies


### 2️⃣ Run the backend


### 3️⃣ Open the frontend
- Open **check_health.html** in any browser  
- Enter patient lab details  
- Get prediction + health recommendations  
- Download PDF report  

---

## 📈 Machine Learning Workflow

1. Imported and cleaned thyroid dataset  
2. Performed feature selection and preprocessing  
3. Trained multiple ML models  
4. Compared accuracy (Random Forest performed best)  
5. Integrated the model into Flask backend  
6. Built frontend pages for users  
7. Added PDF report generation  

---

## 🎯 Project Goal

To provide a fast, accurate, and user-friendly thyroid health assessment system using machine learning, helping individuals and healthcare providers make informed decisions.

---

## 📝 Future Improvements
- Deploy project on cloud (Render / AWS / Heroku)  
- Add patient login and history tracking  
- Integrate SMS/Email notifications  
- Add deep learning model for improved accuracy  
- Support thyroid scan image analysis  

---

## ❤️ About This Project

This project was built as part of an AI & Data Science learning journey, combining machine learning, web development, and healthcare insights to create a complete prediction + reporting platform.

---

## 📊 Model Performance

The dataset was trained using multiple machine learning models and evaluated using accuracy, precision, recall, and confusion matrix.

| Model                | Accuracy |
|---------------------|----------|
| Random Forest       | 96%      |
| SVM                 | 92%      |
| Logistic Regression | 88%      |
| Decision Tree       | 90%      |

✔ **Random Forest** gave the best and most stable performance.

---

## 🛠 Tech Stack

### **Languages**
- Python
- HTML
- CSS
- JavaScript

### **Libraries & Frameworks**
- Flask  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- ReportLab  

### **Tools**
- VS Code  
- GitHub  
- Python 3.x  

---

## 🖼 Screenshots (Optional but Recommended)

You can upload screenshots to a folder named `images/` and show them here.

#prediction result

_(Upload screenshots later into an **images** folder and they will appear automatically.)_

---

## ⭐ Final Notes

This project demonstrates:
- End-to-end ML workflow  
- Web development + backend integration  
- Healthcare-focused machine learning  
- Clean UI + PDF reporting  

It is ideal for showcasing skills in **AI, ML, Python, Flask, and Full-Stack integration.**

---

##Badges Section
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Machine Learning](https://img.shields.io/badge/ML-Healthcare-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

#Installation Guide
## ⚙️ Installation

Follow these steps to set up the project locally:

1️⃣ Clone the repository  
git clone https://github.com/Megha385/Thyroid-disease-prediction

2️⃣ Navigate to the project folder  
cd Thyroid-disease-prediction

3️⃣ Install dependencies  
pip install -r requirements.txt

4️⃣ Run the backend  
python app.py


---

## ✨ Author
**Megha Bandi**  
Final Year – AI & Data Science  
Thyroid Health Prediction | ML | Flask | Data Science

Feel free to connect with me on GitHub! 😊
