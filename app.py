import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

FEATURES = {
    'Priceperweek': {
        'label': 'Price per week',
        'min': 0,
        'max': 1000,
        'placeholder': 'e.g. 30',
    },
    'Population': {
        'label': 'City population',
        'min': 0,
        'max': 10000000,
        'placeholder': 'e.g. 1700000',
    },
    'Monthlyincome': {
        'label': 'Average monthly income',
        'min': 0,
        'max': 100000,
        'placeholder': 'e.g. 12000',
    },
    'Averageparkingpermonth': {
        'label': 'Average parking per month',
        'min': 0,
        'max': 1000,
        'placeholder': 'e.g. 80',
    },
}

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=FEATURES,
        submitted={},
        result=None,
        error=None,
        metrics=None,
    )

@app.route('/predict', methods=['POST'])
def predict():
    submitted = {name: request.form.get(name, '') for name in FEATURES}
    try:
        input_values = [float(request.form.get(name, '')) for name in FEATURES]
        final_features = np.array([input_values])
        prediction = model.predict(final_features)
        output = round(float(prediction[0]), 2)
        result = {'value': output}
        return render_template(
            'index.html',
            features=FEATURES,
            submitted=submitted,
            result=result,
            error=None,
            metrics=None,
        )
    except ValueError:
        return render_template(
            'index.html',
            features=FEATURES,
            submitted=submitted,
            result=None,
            error='Please enter valid numbers for all fields.',
            metrics=None,
        )
    except Exception:
        return render_template(
            'index.html',
            features=FEATURES,
            submitted=submitted,
            result=None,
            error='Prediction failed. Please try again.',
            metrics=None,
        )

if __name__ == '__main__':
    app.run(debug=True)
