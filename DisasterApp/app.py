from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

loaded_model = tf.keras.models.load_model("DisasterTFModel")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    pred_prob = loaded_model.predict([sentence])
    pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
    prediction = "Real Disaster" if pred_label > 0 else "Not Real Disaster"
    probability = pred_prob[0][0]
    return render_template('result.html', sentence=sentence, prediction=prediction, probability=probability)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/analysis')
def analysis():
    # Mock data analysis
    total_questions = 120
    total_real_disasters = 39
    total_not_real_disasters = total_questions - total_real_disasters

    data = pd.DataFrame({
        'Label': ['Real Disasters', 'Not Real Disasters'],
        'Count': [total_real_disasters, total_not_real_disasters]
    })

    plt.figure(figsize=(6, 6))
    plt.title('Disaster Type Distribution')
    plt.pie(data['Count'], labels=data['Label'], autopct='%1.1f%%')
    plt.savefig('static/images/pie_chart.png')

    return render_template('analysis.html', data={
        'total_real_disasters': total_real_disasters,
        'total_not_real_disasters': total_not_real_disasters
    })


if __name__ == '__main__':
    app.run(debug=True)
