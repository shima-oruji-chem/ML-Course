# Diabetes Prediction API for ML Course
## API Input
The API works based on a json input, which has mandatory and optional elements. The mandatory elements are the most important features based on our correlation analysis and are required to make a reliable prediction. These elements are shown below as:

```
{
    "Glucose": "smith",
    "BMI": "john",
    "Age": 25
}
```

A complete json input with all the mandatory and optional elements is as follows:

```
{
    "Pregnancies": 2,
    "Glucose": 148,	   
    "BloodPressure": 72,
    "SkinThickness": 35,	
    "Insulin": 88,	
    "BMI": 29,
    "DiabetesPedigreeFunction": 0.236,
    "Age": 35
}

```

## API Output
If you don't enter a mandatory input (e.g., Glucose), you will receive the following response from the server: 

```
{
  "response": "Mandatory input (Glucose) is missing"
}
```
If you make a proper request to the API, you will get a response similar to: 
```
{
  "diabetes_probability": 0.75
}
```

## API Testing
You can test the API by running the `python3 app.py` in the terminal and making a sample request from another terminal as: 
```
curl http://localhost:5444/diabetes_prediction -H "Content-Type: application/json" -d '{"Pregnancies": 2,"Glucose": 148,"BloodPressure": 72,"SkinThickness": 35,"Insulin": 88,"BMI": 29,"DiabetesPedigreeFunction": 0.236,"Age": 35}'
```

You should receive a response in the json format. If not, there is bug in either your request or in the API program. Here are two other sample requests with which you can test the API:
```
curl http://localhost:5444/diabetes_prediction -H "Content-Type: application/json" -d '{"Pregnancies": 2,"Glucose": 148,"BloodPressure": 72,"SkinThickness": 35,"Insulin": 88,"BMI": 29,"Age": 35}'
```
```
curl http://localhost:5444/diabetes_prediction -H "Content-Type: application/json" -d '{"Pregnancies": 2,"BloodPressure": 72,"SkinThickness": 35,"Insulin": 88,"BMI": 29,"DiabetesPedigreeFunction": 0.236,"Age": 35}'
```