from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "medical_key_2026"

# Configuration Database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="skin_cancer_db"
)

# Chargement du Modèle AI
MODEL_PATH = 'model/vgg16_malignant_vs_bengin.h5'
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            return redirect(url_for('dashboard'))
        else:
            flash("Identifiants incorrects !", "danger")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session: return redirect(url_for('login'))
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM patients")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM patients WHERE result='Malignant'")
    m_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM patients WHERE result='Benign'")
    b_count = cursor.fetchone()[0]
    return render_template('dashboard.html', total=total, m_count=m_count, b_count=b_count)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'loggedin' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        p_name = request.form['patient_name']
        if file:
            filename = file.filename
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)
            
            # Prétraitement et Prédiction AI
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            
            pred_raw = model.predict(x)[0][0]
            
            # Correction de l'erreur numpy.float32 pour MySQL
            res = "Malignant" if pred_raw > 0.5 else "Benign"
            conf = float(pred_raw * 100) if pred_raw > 0.5 else float((1 - pred_raw) * 100)
            
            # Insertion dans la base de données
            cursor = db.cursor()
            query = "INSERT INTO patients (name, result, confidence, image_path, date_added) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (p_name, res, round(conf, 2), filename, datetime.now()))
            db.commit()
            
            # Récupérer l'ID pour afficher le résultat spécifique
            last_id = cursor.lastrowid
            return redirect(url_for('result', report_id=last_id))
            
    return render_template('predict.html')

@app.route('/result/<int:report_id>')
def result(report_id):
    if 'loggedin' not in session: return redirect(url_for('login'))
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients WHERE id = %s", (report_id,))
    report = cursor.fetchone()
    return render_template('result.html', report=report)

@app.route('/patients')
def patients():
    if 'loggedin' not in session: return redirect(url_for('login'))
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients ORDER BY date_added DESC")
    all_patients = cursor.fetchall()
    return render_template('patients.html', patients=all_patients)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)