from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask('DATA_VISUAL')
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("naivebayesmodel.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def ProbPredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("naivebayesmodel.pkl", "rb"))
    probability = loaded_model.predict_proba(to_predict)[:,-1]
    return probability

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)  
        if int(result)== 1:
            prediction ='CONGRATS!!You will be Placed,  probability of You getting placed is:'
        else:
            prediction ='Need to Work hard! You will Not be Placed,  probability of You getting placed is:'
        probability = ProbPredictor(to_predict_list)
        return render_template("predictorform.html", prediction = prediction ,probability=probability*100)




        

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')


if __name__ == "__main__":
    app.run(debug=True)        