import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Durasi Pengiriman Paket",
    page_icon="📦",
    layout="wide"
)

# Custom CSS untuk tampilan lebih menarik
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📦 Prediksi Durasi Pengiriman Paket</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Menggunakan Algoritma Decision Tree Regression</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4829/4829468.png", width=150)
    st.title("⚙️ Pengaturan")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    st.markdown("---")
    
    # Parameter Decision Tree
    st.subheader("Parameter Decision Tree")
    max_depth = st.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Informasi")
    st.info("Tugas UAS Machine Learning\n\nPrediksi Durasi Pengiriman Paket")

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil diupload!")
else:
    st.warning("⚠️ Silakan upload dataset pengiriman.csv")
    st.stop()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Model Training", "🎯 Prediksi", "📈 Visualisasi"])

# Tab 1: Dataset
with tab1:
    st.header("Dataset Pengiriman Paket")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", len(df))
    with col2:
        st.metric("Jumlah Fitur", len(df.columns) - 1)
    with col3:
        st.metric("Target", "Durasi (jam)")
    
    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tipe Data:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

# Tab 2: Model Training
with tab2:
    st.header("Training Model Decision Tree")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Preprocessing
            df_encoded = df.copy()
            label_encoders = {}
            
            # Encode categorical features
            categorical_cols = ['Layanan', 'Cuaca', 'Wilayah']
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le
            
            # Split features and target
            X = df_encoded.drop('Durasi', axis=1)
            y = df_encoded['Durasi']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['label_encoders'] = label_encoders
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred_train'] = y_pred_train
            st.session_state['y_pred_test'] = y_pred_test
            st.session_state['feature_names'] = X.columns.tolist()
            
        st.success("✅ Model berhasil ditraining!")
    
    # Display results if model exists
    if 'model' in st.session_state:
        st.subheader("Evaluasi Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Training Set")
            mae_train = mean_absolute_error(st.session_state['y_train'], st.session_state['y_pred_train'])
            rmse_train = np.sqrt(mean_squared_error(st.session_state['y_train'], st.session_state['y_pred_train']))
            r2_train = r2_score(st.session_state['y_train'], st.session_state['y_pred_train'])
            
            st.metric("MAE", f"{mae_train:.2f} jam")
            st.metric("RMSE", f"{rmse_train:.2f} jam")
            st.metric("R² Score", f"{r2_train:.4f}")
        
        with col2:
            st.markdown("### 📊 Testing Set")
            mae_test = mean_absolute_error(st.session_state['y_test'], st.session_state['y_pred_test'])
            rmse_test = np.sqrt(mean_squared_error(st.session_state['y_test'], st.session_state['y_pred_test']))
            r2_test = r2_score(st.session_state['y_test'], st.session_state['y_pred_test'])
            
            st.metric("MAE", f"{mae_test:.2f} jam")
            st.metric("RMSE", f"{rmse_test:.2f} jam")
            st.metric("R² Score", f"{r2_test:.4f}")
        
        # Feature Importance
        st.subheader("Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': st.session_state['feature_names'],
            'Importance': st.session_state['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feature_imp, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)

# Tab 3: Prediksi
with tab3:
    st.header("Prediksi Durasi Pengiriman")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Model Training'")
    else:
        # Initialize prediction history if not exists
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []
        
        # Initialize reset counter for forcing widget refresh
        if 'reset_counter' not in st.session_state:
            st.session_state['reset_counter'] = 0
        
        st.subheader("Masukkan Data Pengiriman")
        
        # Use form to group inputs together
        with st.form("prediction_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                jarak = st.number_input("Jarak (km)", min_value=0, max_value=1000, value=50, 
                                       key=f'jarak_{st.session_state["reset_counter"]}')
                berat = st.number_input("Berat (kg)", min_value=0, max_value=100, value=5,
                                       key=f'berat_{st.session_state["reset_counter"]}')
                layanan = st.selectbox("Layanan", df['Layanan'].unique(),
                                      key=f'layanan_{st.session_state["reset_counter"]}')
            
            with col2:
                cuaca = st.selectbox("Cuaca", df['Cuaca'].unique(),
                                    key=f'cuaca_{st.session_state["reset_counter"]}')
                wilayah = st.selectbox("Wilayah", df['Wilayah'].unique(),
                                      key=f'wilayah_{st.session_state["reset_counter"]}')
            
            # Submit button inside form
            predict_button = st.form_submit_button("🎯 Prediksi Durasi", type="primary", use_container_width=True)
        
        # Buttons outside form for other actions
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            reset_button = st.button("🔄 Reset Input", use_container_width=True)
        
        with col_btn2:
            clear_history = st.button("🗑️ Hapus History Prediksi", use_container_width=True)
        
        # Handle reset button
        if reset_button:
            st.session_state['reset_counter'] += 1
            st.success("✅ Input berhasil direset!")
            st.rerun()
        
        # Handle clear history button
        if clear_history:
            st.session_state['prediction_history'] = []
            st.success("✅ History prediksi berhasil dihapus!")
        
        # Handle prediction
        if predict_button:
            # Encode input
            layanan_encoded = st.session_state['label_encoders']['Layanan'].transform([layanan])[0]
            cuaca_encoded = st.session_state['label_encoders']['Cuaca'].transform([cuaca])[0]
            wilayah_encoded = st.session_state['label_encoders']['Wilayah'].transform([wilayah])[0]
            
            # Create input array
            input_data = np.array([[jarak, berat, layanan_encoded, cuaca_encoded, wilayah_encoded]])
            
            # Predict
            prediction = st.session_state['model'].predict(input_data)[0]
            
            # Store in history
            from datetime import datetime
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'jarak': jarak,
                'berat': berat,
                'layanan': layanan,
                'cuaca': cuaca,
                'wilayah': wilayah,
                'durasi_prediksi': round(prediction, 1)
            }
            st.session_state['prediction_history'].insert(0, prediction_record)  # Insert at beginning
            
            # Keep only last 10 predictions
            if len(st.session_state['prediction_history']) > 10:
                st.session_state['prediction_history'] = st.session_state['prediction_history'][:10]
            
            # Display result
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durasi Prediksi", f"{prediction:.1f} jam")
            with col2:
                st.metric("Estimasi Hari", f"{prediction/24:.1f} hari")
            with col3:
                if prediction < 24:
                    status = "🟢 Cepat"
                elif prediction < 48:
                    status = "🟡 Sedang"
                else:
                    status = "🔴 Lambat"
                st.metric("Status", status)
            
            # Detail prediksi
            st.info(f"""
            **Detail Pengiriman:**
            - Jarak: {jarak} km
            - Berat: {berat} kg
            - Layanan: {layanan}
            - Cuaca: {cuaca}
            - Wilayah: {wilayah}
            - **Estimasi Durasi: {prediction:.1f} jam ({prediction/24:.1f} hari)**
            """)
        
        # Display prediction history
        if len(st.session_state['prediction_history']) > 0:
            st.markdown("---")
            st.subheader("📋 History Prediksi Terakhir")
            
            # Convert to DataFrame for better display
            history_df = pd.DataFrame(st.session_state['prediction_history'])
            history_df['durasi_hari'] = (history_df['durasi_prediksi'] / 24).round(1)
            
            # Add status column
            def get_status(durasi):
                if durasi < 24:
                    return "🟢 Cepat"
                elif durasi < 48:
                    return "🟡 Sedang"
                else:
                    return "🔴 Lambat"
            
            history_df['status'] = history_df['durasi_prediksi'].apply(get_status)
            
            # Reorder columns
            display_df = history_df[['timestamp', 'jarak', 'berat', 'layanan', 'cuaca', 
                                    'wilayah', 'durasi_prediksi', 'durasi_hari', 'status']]
            
            # Rename columns for better display
            display_df.columns = ['Waktu', 'Jarak (km)', 'Berat (kg)', 'Layanan', 
                                 'Cuaca', 'Wilayah', 'Durasi (jam)', 'Durasi (hari)', 'Status']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# Tab 4: Visualisasi
with tab4:
    st.header("Visualisasi Decision Tree")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Model Training'")
    else:
        st.subheader("Struktur Decision Tree")
        
        # Plot decision tree
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            st.session_state['model'],
            feature_names=st.session_state['feature_names'],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Actual vs Predicted
        st.subheader("Actual vs Predicted Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Set")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state['y_train'], st.session_state['y_pred_train'], alpha=0.5)
            ax.plot([st.session_state['y_train'].min(), st.session_state['y_train'].max()], 
                    [st.session_state['y_train'].min(), st.session_state['y_train'].max()], 
                    'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Training Set: Actual vs Predicted')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Testing Set")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state['y_test'], st.session_state['y_pred_test'], alpha=0.5, color='green')
            ax.plot([st.session_state['y_test'].min(), st.session_state['y_test'].max()], 
                    [st.session_state['y_test'].min(), st.session_state['y_test'].max()], 
                    'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Testing Set: Actual vs Predicted')
            st.pyplot(fig)
        
        # Residual plot
        st.subheader("Residual Plot")
        residuals = st.session_state['y_test'] - st.session_state['y_pred_test']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(st.session_state['y_pred_test'], residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📦 Aplikasi Prediksi Durasi Pengiriman Paket | Tugas UAS Machine Learning</p>
    </div>
""", unsafe_allow_html=True)