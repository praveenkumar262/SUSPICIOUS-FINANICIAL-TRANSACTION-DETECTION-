from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
 
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
 
# Load the trained model and preprocessing objects
model = joblib.load("best_suspicious_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
 
# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            return "Username already exists!"
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('predict'))
        return "Invalid credentials!"
    return render_template('login.html')
 
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))
 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
   
    prediction = None  # Initialize prediction here to avoid UnboundLocalError
 
    if request.method == 'POST':
        try:
            data = [float(request.form[key]) for key in request.form.keys()]
            scaled_data = scaler.transform([data])
            prediction = model.predict(scaled_data)[0]
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            # Return plain text instead of JSON
            return f"Prediction Result: {prediction_label}"
        except Exception as e:
            return str(e)
 
    # For GET request, pass None (or a default value) to the template
    return render_template("predict.html", prediction=prediction)
 
# ✅ THIS IS THE CORRECT PLACE TO CREATE TABLES IN FLASK 3.x
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)