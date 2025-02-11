import os
import gc
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from constants import symptoms_dict, diseases_list, verbose_name
from dotenv import load_dotenv
# Disable OneDNN optimizations to reduce memory usage
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load SVM model once (lightweight)
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Lazy Load TensorFlow Model
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("models/skin.h5", compile=False)
    return model

def unload_model():
    """Unload model to free memory after each request."""
    global model
    if model is not None:
        del model
        gc.collect()
        tf.keras.backend.clear_session()

def load_csv(filename):
    """Load CSV files only when needed to reduce memory usage."""
    return pd.read_csv(f"datasets/{filename}")

def predict_label(img_path):
    """Predict disease from image using the TensorFlow model."""
    model = get_model()
    
    test_image = image.load_img(img_path, target_size=(28, 28))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 28, 28, 3)

    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    
    unload_model()  # Free memory after prediction
    
    return [verbose_name[classes_x[0]], predict_x, classes_x]

def helper(dis):
    """Get disease details dynamically without keeping all CSVs in memory."""
    description = load_csv("description.csv")
    precautions = load_csv("precautions_df.csv")
    medications = load_csv("medications.csv")
    diets = load_csv("diets.csv")
    workout = load_csv("workout_df.csv")
    
    desc = " ".join(description[description['Disease'] == dis]['Description'].astype(str))
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()
    
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    """Predict disease based on user symptoms."""
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    
    return diseases_list[svc.predict(input_df)[0]]


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the AI-Powered Disease Prediction API!",
        "routes": {
            "/predict": "POST - Predict disease based on symptoms",
            "/predictdisease": "POST - Predict skin disease from image"
        },
        "status": "API is running successfully 🚀"
    })



@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for symptom-based disease prediction."""
    try:
        data = request.json
        symptoms = data.get('symptoms', '')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
        my_precautions = [i for i in precautions[0] if i != 'nan']

        return jsonify({
            'predicted_disease': str(predicted_disease),
            'medications': list(medications),
            'dis_des': str(dis_des),
            'my_precautions': my_precautions,
            'my_diet': list(rec_diet),
            'workout': list(workout)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/predictdisease", methods=['POST'])
def get_output():
    """API endpoint for image-based disease prediction."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        img = request.files['image']
        img_path = f"static/tests/{img.filename}"
        img.save(img_path)

        predict_result = predict_label(img_path)
        index = predict_result[2][0]
        max_value = predict_result[1][0][index]

        if max_value < 0.75 or max_value == 1.0:
            return jsonify({"disease": False})

        return jsonify({"disease": predict_result[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    app.run(debug=debug, port=port)
