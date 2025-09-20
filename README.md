# Eudaimonia: Your Confidential AI Wellness Companion

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Eudaimonia is an AI-powered wellness solution that detects the stress-sleep-eating cycle in Indian students, offering confidential, empathetic, and proactive support.**

This project was developed for the Gen AI Exchange Hackathon.

---

##  Live Demo

**[Link to the Deployed Application]** - *https://eudaimonia.up.railway.app/*

---

## Table of Contents
1. [The Problem](#the-problem)
2. [Our Solution](#our-solution)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [System Architecture](#system-architecture)
6. [Local Setup and Installation](#local-setup-and-installation)
7. [Project Structure](#project-structure)
8. [Team](#team)
9. [License](#license)

---

## The Problem

Indian students face immense academic and social pressures, leading to a silent crisis of stress, anxiety, and related health issues. The fear of judgment and stigma creates a major barrier, preventing them from seeking the help they need. This results in a negative cycle of poor sleep, irregular eating, and deteriorating mental well-being.

## Our Solution

**Eudaimonia** (from the Greek for "human flourishing") is a confidential, AI-powered mental wellness solution designed to support and guide Indian youth. We provide a safe, anonymous space where students can understand their mental state without fear. By analyzing the interconnected cycle of stress, sleep, and eating, Eudaimonia offers empathetic, data-driven insights, helping students overcome stigma and take the first step towards accessing help.

## Key Features

* **Anonymous Onboarding:** A private entry with a consent form. No personal data or login is required.
* **Wellness Assessment:** A quick quiz on stress (PSS-10), sleep (SCI), and eating habits.
* **AI Pattern Analysis:** A custom TensorFlow model finds the unique connection between a user's stress, sleep, and eating.
* **Empathetic Feedback:** Uses Google's Gemini API to generate an instant, easy-to-understand wellness report.
* **Crisis Support:** One-click, direct access to verified mental health helplines.

## Technology Stack

The solution is built using a modern, robust technology stack:

* **Backend:** Python 3.10, Flask, Gunicorn
* **Machine Learning / AI:** TensorFlow, Keras, Scikit-learn, Pandas, Joblib, Google Gemini API
* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
* **Deployment:** Render (or similar cloud platform)
* **Version Control:** Git & GitHub

## System Architecture

The application follows a simple, scalable architecture:
1.  **Frontend:** The user interacts with the web interface built with HTML/CSS/JS.
2.  **Backend (Flask API):** The frontend sends anonymous questionnaire data to the Flask server.
3.  **Data Processing:** The backend preprocesses the data using the saved Scikit-learn scaler.
4.  **AI/ML Model:** The preprocessed data is fed into the trained TensorFlow/Keras model for prediction. The results are also sent to the Google Gemini API to generate empathetic text.
5.  **Response:** The final insights are sent back to the frontend and displayed to the user.

## Local Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [repository-name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Project Structure

├── app.py              # Main Flask application logic
├── youth_wellness_model.keras # Trained TensorFlow/Keras model
├── scaler.joblib       # Saved Scikit-learn scaler
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment configuration for Render/Heroku
├── static/             # Contains CSS, JS, images
└── templates/          # Contains HTML files for the frontend

## To run locally pull website.zip

## Team

* * **Chaitanya A S** *
* * **Gauravi k S** *
* * **Aneesh  Aditya SR** *


## License

This project is licensed under the **MIT License**.
