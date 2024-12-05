import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Download and Load Data
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
history = mlp_cl.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Evaluate the Model
loss, accuracy = mlp_cl.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Making Predictions
predictions = mlp_cl.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)
print("Predicted classes:", predicted_classes.flatten())


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, predicted_classes)
print("Classification Report:\n", class_report)

