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

st.set_page_config(page_title="Platformă Enterprise Predicție Churn SaaS", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2ec4b6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

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

        joblib.dump({'model': model, 'encoders': encoders, 'features': features}, model_path)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    model_data = joblib.load(model_path)
    return df, model_data

df, model_data = setup_data_and_model()

st.sidebar.title("🔍 Filtre Dashboard")
contract_filter = st.sidebar.multiselect("Tip Contract", options=df["Contract"].unique(), default=df["Contract"].unique())
internet_filter = st.sidebar.multiselect("Serviciu Internet", options=df["InternetService"].unique(), default=df["InternetService"].unique())

filtered_df = df[(df["Contract"].isin(contract_filter)) & (df["InternetService"].isin(internet_filter))]

st.title("Platformă de Analiză și Prevenire a Churn-ului")
tabs = st.tabs(["📈 Executive Dashboard", "🔮 AI Churn Predictor & What-If", "➕ Adăugare Client (CRUD)", "🗄️ Bază de Date & Export"])

with tabs[0]:
    st.header("Metrice & KPI-uri Financiare")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clienți Filtrați", len(filtered_df))
    col2.metric("Rată Churn Medie", f"{(filtered_df['Churn'].mean() * 100):.1f}%")
    col3.metric("Venit Mediu Lunar", f"${filtered_df['MonthlyCharges'].mean():.2f}")
    col4.metric("Valoare Totală Portofoliu", f"${filtered_df['TotalCharges'].sum():,.0f}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribuția Churn-ului pe Contracte")
        fig_contract = px.histogram(filtered_df, x="Contract", color="Churn", barmode="group",
                                   color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
                                   labels={'Churn': 'Client Anulat (1=Da)'})
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with c2:
        st.subheader("Metodă de Plată vs Riscul de Churn")
        fig_pay = px.histogram(filtered_df, x="PaymentMethod", color="Churn", barmode="stack",
                               color_discrete_map={0: "#2ec4b6", 1: "#e71d36"})
        st.plotly_chart(fig_pay, use_container_width=True)

with tabs[1]:
    st.header("Simulator de Risc Client & Strategie Retenție")
    
    col_a, col_b = st.columns(2)
    with col_a:
        tenure = st.number_input("Vechime (luni)", min_value=1, max_value=100, value=6)
        monthly_charges = st.number_input("Abonament Lunar ($)", min_value=10.0, max_value=200.0, value=85.0)
        total_charges = tenure * monthly_charges
        st.caption(f"Estimare Încasări Totale (CLV): **${total_charges:.2f}**")

    with col_b:
        contract = st.selectbox("Tip Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Serviciu Internet", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Metodă Plată", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.button("🚀 Calculează Riscul & Analizează Factorii", type="primary"):
        encoders = model_data['encoders']
        model = model_data['model']
        
        input_df = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': encoders['Contract'].transform([contract])[0],
            'InternetService': encoders['InternetService'].transform([internet])[0],
            'PaymentMethod': encoders['PaymentMethod'].transform([payment])[0]
        }])
        
        prob = model.predict_proba(input_df)[0][1] * 100
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("Rezultat Evaluare")
            if prob > 50:
                st.error(f"⚠️ **RISC RIDICAT:** {prob:.1f}% probabilitate de Churn")
            else:
                st.success(f"✅ **RISC SCĂZUT:** {prob:.1f}% probabilitate de Churn")
                
            st.markdown("---")
            st.markdown("💡 **Simulare de Retenție (What-If):**")
            if contract == "Month-to-month":
                sim_input = input_df.copy()
                sim_input['Contract'] = encoders['Contract'].transform(["One year"])[0]
                sim_prob = model.predict_proba(sim_input)[0][1] * 100
                diff = prob - sim_prob
                st.info(f"Dacă convingi clientul să treacă la **Contract pe 1 An**, riscul scade cu **{diff:.1f}%** (ajunge la {sim_prob:.1f}%).")
            else:
                st.write("Clientul are deja un contract pe termen lung. Risc scăzut de anulare spontană.")

        with res_col2:
            st.subheader("Importanța Factorilor în Model")
            importances = model.feature_importances_
            feature_names = ['Tenure', 'Monthly Charges', 'Total Charges', 'Contract', 'Internet', 'Payment Method']
            imp_df = pd.DataFrame({'Factor': feature_names, 'Importanță': importances}).sort_values(by='Importanță', ascending=True)
            
            fig_imp = px.bar(imp_df, x='Importanță', y='Factor', orientation='h', color='Importanță', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

with tabs[2]:
    st.header("➕ Înregistrare Client Nou în Baza de Date SQL")
    
    with st.form("new_customer_form"):
        f_id = st.text_input("Customer ID (Ex: 9999-NEW)", value="1000-NEW")
        f_tenure = st.number_input("Vechime (luni)", min_value=0, max_value=120, value=1)
        f_monthly = st.number_input("Abonament Lunar ($)", min_value=10.0, max_value=200.0, value=50.0)
        f_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        f_internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        f_payment = st.selectbox("Plată", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        submitted = st.form_submit_button("💾 Salvează în Baza de Date SQL")
        
        if submitted:
            db_path = os.path.join('database', 'saas_database.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            total_c = f_tenure * f_monthly
            
            cursor.execute("""
                INSERT INTO customers (customerID, tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, Churn)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (f_id, f_tenure, f_monthly, total_c, f_contract, f_internet, f_payment))
            
            conn.commit()
            conn.close()
            st.success(f"✅ Clientul {f_id} a fost adăugat cu succes în baza de date SQL!")
            st.cache_resource.clear()

with tabs[3]:
    st.header("🗄️ Bază de Date & Export Rapoarte")
    
    st.dataframe(filtered_df, use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descarcă Raportul Filtrat în CSV",
        data=csv_data,
        file_name='raport_clienti_churn.csv',
        mime='text/csv',
        type="primary"
    )
