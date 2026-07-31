import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="SaaS Churn & Analytics", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'churn_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def get_data():
    db_path = os.path.join('database', 'saas_database.db')
    csv_path = os.path.join('data', 'customers.csv')
    
    if not os.path.exists(db_path):
        if os.path.exists(csv_path):
            os.makedirs('database', exist_ok=True)
            df_raw = pd.read_csv(csv_path)
            df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'].replace(' ', None), errors='coerce')
            df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(df_raw['tenure'] * df_raw['MonthlyCharges'])
            df_raw['Churn'] = df_raw['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
            conn = sqlite3.connect(db_path)
            df_raw.to_sql('customers', conn, if_exists='replace', index=False)
            conn.close()
        else:
            return None

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    return df


st.title("📊 SaaS Customer Churn & Business Analytics")
st.markdown("Proiect Hibrid: **Informatică Economică (SQL/Web)** + **Cibernetică (ML/Analytics)**")

df = get_data()
model_data = load_model()

if df is None:
    st.error("Baza de date nu a fost găsită! Rulează mai întâi `python database/db_builder.py` în terminal.")
else:
    # Creare Tab-uri pentru aplicație
    tabs = st.tabs(["📈 Business Dashboard", "🔮 Churn Predictor", "🗄️ Data Management"])

# TAB 1
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

# TAB 2
    with tabs[1]:
        st.header("Simulator de Risc Client (Machine Learning)")
        
        if model_data is None:
            st.error("Modelul ML nu a fost găsit! Rulează `python models/train_model.py` mai întâi.")
        else:
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

# TAB 3
    with tabs[2]:
        st.header("Vizualizare Bază de Date (SQL)")
        st.dataframe(df.head(100), use_container_width=True)
