# Importing all the Required Libraries
import pandas as pd
import joblib

def predict(data):
    mmscaler = joblib.load('min_max_scaler.pkl')

    # Lets see what is in the Data

    df = pd.DataFrame(data)

    df.groupby(['airline','source_city','destination_city'],as_index=False)['price'].mean().head(10)

    # Creating a Back up File
    df_bk=df.copy()

    # Coverting the labels into a numeric form using Label Encoder
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    for col in df.columns:
        if df[col].dtype=='object':
            le = joblib.load( f'label_encoder{col}.pkl')
            df[col]=le.transform(df[col])

    # Scaling the values to convert the int values to Machine Languages
    from sklearn.preprocessing import MinMaxScaler
    x=mmscaler.transform(df)
    x=pd.DataFrame(df)

    from sklearn.ensemble import ExtraTreesRegressor

    modelETR = joblib.load('extra_trees_regressor_model.pkl')

    y_pred = modelETR.predict(x)

    return y_pred