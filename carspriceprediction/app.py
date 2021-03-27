from flask import Flask, render_template, request
import joblib
import os
from datetime import date
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load('RandomFRegressor.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

standard_to = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    fuel_type_diesel=0
    if request.method == 'POST':
        current_date = date.today()
        year = int(request.form['Year'])
        year = (current_date.year - year)
        present_price=float(request.form['Present_Price'])
        kms_driven=int(request.form['Kms_Driven'])
        owner=int(request.form['Owner'])

        fuel_type_petrol=request.form['Fuel_Type_Petrol']
        if(fuel_type_petrol=='Petrol'):
                fuel_type_petrol=1
                fuel_type_diesel=0
        else:
            fuel_type_petrol=0
            fuel_type_diesel=1

        seller_type_individual=request.form['Seller_Type_Individual']
        if(seller_type_individual=='Individual'):
            seller_type_individual=1
        else:
            seller_type_individual=0

        transmission_mannual=request.form['Transmission_Mannual']
        if(transmission_mannual=='Mannual'):
            transmission_mannual=1
        else:
            transmission_mannual=0

        prediction = model.predict([[present_price,kms_driven,owner,year,fuel_type_diesel,fuel_type_petrol,seller_type_individual,transmission_mannual]])
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