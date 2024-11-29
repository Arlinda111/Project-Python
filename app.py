import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model1.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = int(request.form['age'])
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']

        arr = [[age, str(hypertension), str(heart_disease)]]
        pred = model.predict(arr)

        prediction_mapping = {'Yes': 'Not Normal', 'No': 'Normal'}
        prediction_label = prediction_mapping[pred[0]]

        return render_template('home.html', prediction=prediction_label)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
