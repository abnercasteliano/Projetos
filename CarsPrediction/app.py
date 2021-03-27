from flask import Flask, render_template, request
import joblib
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load('RandomFRegressor.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

standard_to = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Year = (2021 - Year)
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Owner=int(request.form['Owner'])

        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1

        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0

        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0

        prediction = model.predict([[Present_Price,Kms_Driven,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        output=round(prediction[0],2)

        if output <= 0:
            return render_template('index.html', prediction_texts='Lamentamos! Este veículo não pode ser vendido.')
        else:
            return render_template('index.html', prediction_texts='Este veículo pode ser vendido por US$ {0:.2f}'.format(output))
        
    else:
        return render_template('index.html')
        

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)