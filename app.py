import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import io
from constants import symptoms_dict, diseases_list, verbose_name, disease_labels

model = load_model("models/skin.h5", compile=False)
svc = pickle.load(open('models/svc.pkl','rb'))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv") 

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(28,28))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 28,28,3)
	predict_x=model.predict(test_image)     
	classes_x=np.argmax(predict_x,axis=1)
	return [verbose_name[classes_x[0]], predict_x, classes_x]

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc.astype(str))      
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()      
    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist()      
    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist()      
    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist()
    return desc, pre, med, die, wrkout


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    return diseases_list[svc.predict(input_df)[0]]


@app.route('/predict', methods=['POST'])
def predict():
    try:        
        data = request.json
        symptoms = data.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]            
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
        my_precautions = []
        for i in precautions[0]:
            if(i != 'nan'):
                my_precautions.append(i)

        
        return jsonify({
            'predicted_disease': str(predicted_disease),  
            'medications': list(medications),  
            'dis_des': str(dis_des),  
            'my_precautions': list(my_precautions),  
            'my_diet': list(rec_diet),  
            'workout': list(workout)  
        })


    except Exception as e:
        print(jsonify({'error': str(e)}))
        return jsonify({'error': str(e)}), 500
    

@app.route("/predictdisease", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['image']
        img_path = "static/tests/" + img.filename    
        img.save(img_path)
        predict_result = predict_label(img_path)
        print(predict_result)        
        index = predict_result[2][0]
        max_value = predict_result[1][0][index]
        print("the value is ", max_value, index)
        if max_value < 0.75 or max_value == 1.0:
            return jsonify({
            "disease": False
        })

        return jsonify({
            "disease": predict_result[0]
        })

 


if __name__ == '__main__':
    app.run(debug=True)



# @app.route("/predictdiseases", methods=["POST"])
# def predict_disease():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     image = Image.open(io.BytesIO(file.read()))

    
#     image = image.resize((28, 28))  
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)     
#     prediction = model.predict(image)
#     print("Model Prediction Output:", prediction)
    
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction)  

    
#     if predicted_class >= len(verbose_name):
#         return jsonify({"error": "Invalid prediction index"}), 500

#     disease = verbose_name[predicted_class]

#     return jsonify({
#         "disease": disease,
#         "confidence": round(float(confidence) * 100, 2),
#         "prediction":str(prediction) 
#     })
