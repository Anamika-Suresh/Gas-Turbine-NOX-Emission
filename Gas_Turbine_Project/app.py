from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import json


app = Flask(__name__)
# Load the trained model

model=pickle.load(open('model.pkl','rb'))
print('model loaded')

scaled = pickle.load(open('scaling.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:
            # Get form inputs
            AT = float(request.form['AT'])
            print("Ambient Temprature:",AT)
            AP = float(request.form['AP'])
            print("Ambient Pressure:",AP)
            AH = float(request.form['AH'])
            print("Ambient Humidity:",AH)
            AFDP = float(request.form['AFDP'])
            print("Air Filter Difference Pressure:",AFDP)
            GTEP = request.form['GTEP']
            print(" Gas Turbine Exhaust Pressure:",GTEP)
            TIT = float(request.form['TIT'])
            print("Turbine Inlet temperature:",TIT)
            TAT = float(request.form['TAT'])
            print( "Turbine After temperature:", TAT)
            TEY = float(request.form['TEY'])
            print( "Turbine Energy Yield:", TEY)
            CDP = float(request.form['CDP'])
            print( "Compressor Discharge Pressure:", CDP)
            CO = float(request.form['CO'])
            print( "Carbon monoxide content:", CO)

            

            # Prepare data 
            details = [AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO]
            print(details)

            data_out=np.array(details).reshape(1,-1)
            print(data_out)
            print(data_out.shape)

           
            data_scaled = scaled.transform(data_out)


            # data_scaled = scaled.transform(data_out)
            # print("Scaled input:", data_scaled)
            
            # Predict car price
            prediction = model.predict(data_scaled)
            #prediction = model.predict(input_features)[0]
            print(prediction)
            #output = jsonify(float(round(prediction[0], 2)))
            #print(output)
            output =  {'result':prediction[0]}
            #return json.dumps(output)

           
            return render_template('index.html', prediction_text=f'NOX emission:{round(prediction[0], 2)}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
