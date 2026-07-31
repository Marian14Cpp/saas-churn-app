import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SaaS Churn & Analytics", page_icon="📊", layout="wide")

@st.cache_resource
def setup_data_and_model():
    os.makedirs('data', exist_ok=True)
    os.makedirs('database', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    csv_path = os.path.join('data', 'customers.csv')
    db_path = os.path.join('database', 'saas_database.db')
    model_path = os.path.join('models', 'churn_model.pkl')

    if not os.path.exists(csv_path):
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        urllib.request.urlretrieve(url, csv_path)

    if not os.path.exists(db_path):
        df_raw = pd.read_csv(csv_path)
        df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'].replace(' ', None), errors='coerce')
        df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(df_raw['tenure'] * df_raw['MonthlyCharges'])
        df_raw['Churn'] = df_raw['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        
        conn = sqlite3.connect(db_path)
        df_raw.to_sql('customers', conn, if_exists='replace', index=False)
        conn.close()

    if not os.path.exists(model_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()

        features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
        X = df[features].copy()
        y = df['Churn']

        encoders = {}
        for col in ['Contract', 'InternetService', 'PaymentMethod']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump({'model': model, 'encoders': encoders}, model_path)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    model_data = joblib.load(model_path)
    return df, model_data

df, model_data = setup_data_and_model()

st.title("SaaS Customer Churn & Business Analytics")

tabs = st.tabs(["📈 Business Dashboard", "🔮 Churn Predictor", "🗄️ Data Management"])

with tabs[0]:
    st.header("Metrice & KPI-uri Economice")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clienți", len(df))
    col2.metric("Rată Churn", f"{(df['Churn'].mean() * 100):.1f}%")
    col3.metric("Venit Mediu Lunar", f"${df['MonthlyCharges'].mean():.2f}")
    col4.metric("Vechime Medie (Luni)", f"{df['tenure'].mean():.1f}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribuția Churn-ului după Tipul de Contract")
        fig_contract = px.histogram(df, x="Contract", color="Churn", barmode="group",
                                   color_discrete_map={0: "#2ec4b6", 1: "#e71d36"})
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with c2:
        st.subheader("Relația dintre Vechime (Tenure) și Încasări")
        fig_scatter = px.scatter(df, x="tenure", y="TotalCharges", color="Churn",
                                color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
                                opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tabs[1]:
    st.header("Simulator de Risc Client (Machine Learning)")
    st.write("Introduceți datele unui client pentru a calcula probabilitatea de Churn:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        tenure = st.number_input("Vechime (luni)", min_value=1, max_value=100, value=12)
        monthly_charges = st.number_input("Abonament Lunar ($)", min_value=10.0, max_value=200.0, value=65.0)
        total_charges = tenure * monthly_charges
        st.info(f"Încasări Totale Estimate (CLV): ${total_charges:.2f}")

    with col_b:
        contract = st.selectbox("Tip Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Serviciu Internet", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Metodă Plată", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.button("🚀 Calculează Riscul de Churn", type="primary"):
        encoders = model_data['encoders']
        input_df = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': encoders['Contract'].transform([contract])[0],
            'InternetService': encoders['InternetService'].transform([internet])[0],
            'PaymentMethod': encoders['PaymentMethod'].transform([payment])[0]
        }])
        
        model = model_data['model']
        prob = model.predict_proba(input_df)[0][1] * 100
        
        st.divider()
        if prob > 50:
            st.error(f"⚠️ **Risc Ridicat de Churn:** {prob:.1f}% probabilitate ca acest client să anuleze abonamentul.")
        else:
            st.success(f"✅ **Risc Scăzut de Churn:** Doar {prob:.1f}% probabilitate de anulare.")

with tabs[2]:
    st.header("Vizualizare Bază de Date (SQL)")
    st.dataframe(df.head(100), use_container_width=True)
