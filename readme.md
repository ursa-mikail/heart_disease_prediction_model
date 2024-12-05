# Heart Disease Prediction Model
## Project Structure
Ensure your project directory is structured as follows:
```
heart_disease_prediction_model/
├── app.py
├── sample_data/
│   └── heart_statlog_cleveland_hungary_final.csv
└── templates/
    ├── index.html
```
### Files and Directories
app.py: The main Flask application file.
sample_data/: Directory containing the dataset.
templates/: Directory containing the HTML templates.
index.html: The main HTML file for rendering the home page

## Running the Flask Application
Open your terminal and navigate to the project directory:

cd /heart_disease_prediction_model
Set the FLASK_APP environment variable:

<pre>
export FLASK_APP=app.py  # On macOS/Linux
set FLASK_APP=app.py   # On Windows
</pre>

Run the Flask application:
```
flask run --port 5001
```

![terminal_out](terminal_out.png)

This should start your Flask application on port 5001. Now, navigate to http://127.0.0.1:5001 in web browser, Flask should correctly render index.html.

![web_ui](web_ui.png)

Using `heart_disease_prediction_model.py`.
![plots](plots.png)

We can also:
<pre>
df = './sample_data/heart_statlog_cleveland_hungary_final.csv'
df = pd.read_csv(df)
df
</pre>

![csv_view](csv_view.png)

```
df.columns

Index(['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
       'fasting blood sugar', 'resting ecg', 'max heart rate',
       'exercise angina', 'oldpeak', 'ST slope', 'target'],
      dtype='object')
```

```
model = mlp_cl
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True)
```

<img src="model.png" alt="model" width="500" height="600">

```
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = mlp_cl.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2, callbacks=[tensorboard_callback])

#!pip install tensorboard
%load_ext tensorboard
%tensorboard --logdir logs
%tensorboard --logdir logs/func
# ref: https://www.tensorflow.org/tensorboard/graphs

#import tensorflow as tf; print(tf.__version__)
#import tensorboard; print(tensorboard.__version__)
```
![tensorboard](tensorboard.png)

