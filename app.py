from flask import Flask, flash, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from preprocess import crop_brain_contour
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import click
from flask.cli import with_appcontext
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Replace with your database URI
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Define the User model for login and registration
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the preprocess and prediction functions for brain tumor detection
best_model = load_model(filepath='cnn-parameters-improvement-23-0.91.model')
IMG_WIDTH, IMG_HEIGHT = (240, 240)

def preprocess_image(image):
    # Preprocess the image to match the model's input shape
    image = crop_brain_contour(image, plot=False)
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    # Preprocess the image and make predictions using the model
    processed_image = preprocess_image(image)
    prediction = best_model.predict(processed_image)

    # Assuming you have a binary classification model, get the probability of class 1
    probability = float(prediction[0][0])

    # Convert probability to percentage
    probability_percentage = probability * 100

    return prediction, probability_percentage

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('upload_image'))

        flash('Invalid username or password', 'error')

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username is already taken and add the new user to the database
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

# Brain tumor detection route
@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction, probability_percentage = predict_image(image)
        result = "Brain Tumor" if prediction > 0.5 else "No Brain Tumor"
        return render_template('index.html', result=result)

    return render_template('index.html')

# API route for brain tumor detection
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False})

    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    prediction, probability_percentage = predict_image(image)
    result = "Brain Tumor" if prediction > 0.5 else "No Brain Tumor"

    return jsonify({'success': True, 'prediction': result, 'probability': probability_percentage})

# Create a new user using CLI
# flask create-user benassi 12345678

@app.cli.command("create-user")
@click.argument("username")
@click.argument("password")
@with_appcontext
def create_user(username, password):
    """
    Create a new user.

    Example usage:
    flask create-user myusername mypassword
    """
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        click.echo("Username already exists. Please choose a different one.")
    else:
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        click.echo("User created successfully!")

# Run the Flask application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
