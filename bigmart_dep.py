from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('bigmart_model.sav', 'rb'))
encoders = pickle.load(open('bigmart_encoders.sav', 'rb'))
#scalar = pickle.load(open('bigmart_scaler.sav', 'rb'))
#rf_model = pickle.load(open('bigmart_rfmodel.sav', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get form data
    data = {}
    data['Item_Identifier'] = request.form.get('Item_Identifier')
    data['Item_Weight'] = request.form.get('Item_Weight')
    data['Item_Fat_Content'] = request.form.get('Item_Fat_Content')  # cat
    data['Item_Visibility'] = request.form.get('Item_Visibility')
    data['Item_Type'] = request.form.get('Item_Type')  # cat
    data['Item_MRP'] = request.form.get('Item_MRP')
    data['Outlet_Identifier'] = request.form.get('Outlet_Identifier')  # drop
    data['Outlet_Establishment_Year'] = request.form.get('Outlet_Establishment_Year')
    data['Outlet_Size'] = request.form.get('Outlet_Size')  # cat
    data['Outlet_Location_Type'] = request.form.get('Outlet_Location_Type')  # cat
    data['Outlet_Type'] = request.form.get('Outlet_Type')  # cat

    df = pd.DataFrame([data])
    
    for i in encoders['Outlet_Location_Type'].categories_[0]:
        df['Outlet_Location_Type' + '_' + i] = 0.0
    df['Outlet_Location_Type' + '_' + df['Outlet_Location_Type']] = 1.0
    df.drop(columns='Outlet_Location_Type', inplace=True)


    for i in encoders['Item_Fat_Content'].categories_[0]:
        df['Item_Fat_Content' + '_' + i] = 0.0
    df['Item_Fat_Content' + '_' + df['Item_Fat_Content']] = 1.0
    df.drop(columns='Item_Fat_Content', inplace=True)


    for i in encoders['Outlet_Size'].categories_[0]:
        df['Outlet_Size' + '_' + i] = 0.0
    df['Outlet_Size' + '_' + df['Outlet_Size']] = 1.0
    df.drop(columns='Outlet_Size', inplace=True)


    for i in encoders['Item_Type'].categories_[0]:
        df['Item_Type' + '_' + i] = 0.0
    df['Item_Type' + '_' + df['Item_Type']] = 1.0
    df.drop(columns='Item_Type', inplace=True)
    

    for i in encoders['Outlet_Type'].categories_[0]:
        df['Outlet_Type' + '_' + i] = 0.0
    df['Outlet_Type' + '_' + df['Outlet_Type']] = 1.0
    df.drop(columns='Outlet_Type', inplace=True)
    
    
    df = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    
    pred = model.predict(df)
    #return render_template('index.html', {(pred[0][0]), 3})
    #return render_template('index.html',predection_text=  )
    print("Prediction:", pred[0])
    return render_template('index.html', predection_text=pred[0])

if __name__ == "__main__":
    app.run(debug=True)
