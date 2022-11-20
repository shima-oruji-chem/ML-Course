import pickle
import numpy as np


def ml_predict(features) -> float:
    '''
    @features = {
        "Pregnancies": 2,
        "Glucose": 148,	   
        "BloodPressure": 72,
        "SkinThickness": 35,	
        "Insulin": 88,	
        "BMI": 29,
        "DiabetesPedigreeFunction": 0.236,
        "Age": 35
    }
    '''
    with open( "ml_model/ml_model", "rb" ) as trained_ml_model:
        model = pickle.load(trained_ml_model)
        input = []
        for key in features:
            input.append(features[key])
        input_array = np.array([input])
        with open( "ml_model/normalizer", "rb" ) as normalizer:
            nl = pickle.load(normalizer)
            input_array_normalized = nl.transform(input_array)
            
            return float(model.predict(input_array_normalized)[0][0])

if __name__=="__main__":
    features = {
        "Pregnancies": 2,
        "Glucose": 148,	   
        "BloodPressure": 72,
        "SkinThickness": 35,	
        "Insulin": 88,	
        "BMI": 29,
        "DiabetesPedigreeFunction": 0.236,
        "Age": 35
    }
    res = ml_predict(features)
    print(res)