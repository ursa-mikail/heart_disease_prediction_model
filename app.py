import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template

# Download and load data
url = 'https://raw.githubusercontent.com/erdenahmet11/Heart-Disease-Prediction/refs/heads/main/heart_statlog_cleveland_hungary_final.csv'
save_dir = './sample_data/'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'heart_statlog_cleveland_hungary_final.csv')

response = requests.get(url)
response.raise_for_status()

with open(save_path, 'wb') as file:
    file.write(response.content)

# Load data into a DataFrame
df = pd.read_csv(save_path)

# Extracting Input and Output Data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP Model
mlp_cl = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

mlp_cl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
mlp_cl.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = mlp_cl.predict(final_features)
    output = (prediction > 0.5).astype(int)
    result = 'Heart Disease' if output[0][0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port to 5001 or any available port
