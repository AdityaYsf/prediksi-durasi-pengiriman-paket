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

# Function to generate default dataset
@st.cache_data
def generate_default_dataset():
    """Generate dataset default yang realistis"""
    np.random.seed(42)
    n_samples = 75
    
    data = {
        'Jarak': [],
        'Berat': [],
        'Layanan': [],
        'Cuaca': [],
        'Durasi': []
    }
    
    layanan_options = ['Reguler', 'Express', 'Same Day']
    cuaca_options = ['Cerah', 'Berawan', 'Hujan']
    
    layanan_factor = {'Reguler': 1.5, 'Express': 1.0, 'Same Day': 0.6}
    cuaca_factor = {'Cerah': 1.0, 'Berawan': 1.15, 'Hujan': 1.4}
    berat_factor = {'ringan': 1.0, 'sedang': 1.1, 'berat': 1.25}
    
    for i in range(n_samples):
        jarak = np.random.randint(10, 201)
        berat = np.random.randint(1, 51)
        layanan = np.random.choice(layanan_options)
        cuaca = np.random.choice(cuaca_options)
        
        if berat < 5:
            berat_cat = 'ringan'
        elif berat <= 15:
            berat_cat = 'sedang'
        else:
            berat_cat = 'berat'
        
        base_duration = (jarak / 40) * 24
        duration = base_duration
        duration *= layanan_factor[layanan]
        duration *= cuaca_factor[cuaca]
        duration *= berat_factor[berat_cat]
        
        noise = np.random.uniform(0.85, 1.15)
        duration *= noise
        
        if layanan == 'Same Day':
            duration = max(duration, 3)
        elif layanan == 'Express':
            duration = max(duration, 8)
        else:
            duration = max(duration, 12)
        
        duration = round(duration, 1)
        
        data['Jarak'].append(jarak)
        data['Berat'].append(berat)
        data['Layanan'].append(layanan)
        data['Cuaca'].append(cuaca)
        data['Durasi'].append(duration)
    
    return pd.DataFrame(data)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4829/4829468.png", width=150)
    st.title("⚙️ Pengaturan")
    
    # Pilihan sumber dataset
    st.subheader("📊 Sumber Dataset")
    dataset_source = st.radio(
        "Pilih sumber dataset:",
        ["Dataset Default", "Upload CSV"],
        help="Dataset Default: Menggunakan data sample yang sudah tersedia\nUpload CSV: Upload dataset kamu sendiri"
    )
    
    uploaded_file = None
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    st.markdown("---")
    
    # Parameter Decision Tree
    st.subheader("Parameter Decision Tree")
    max_depth = st.slider("Max Depth", 1, 20, 6)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2)
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Informasi")
    st.info("Tugas UAS Machine Learning\n\nPrediksi Durasi Pengiriman Paket")

# Load dataset
if dataset_source == "Dataset Default":
    df = generate_default_dataset()
    st.info("📊 Menggunakan dataset default (75 data pengiriman)")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil diupload!")
else:
    st.warning("⚠️ Silakan upload dataset pengiriman.csv")
    st.stop()

# Validasi kolom dataset
expected_columns = ['Jarak', 'Berat', 'Layanan', 'Cuaca', 'Durasi']
if not all(col in df.columns for col in expected_columns):
    st.error(f"❌ Dataset harus memiliki kolom: {', '.join(expected_columns)}")
    st.stop()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Model Training", "🎯 Prediksi", "📈 Visualisasi"])

# Tab 1: Dataset
with tab1:
    st.header("Dataset Pengiriman Paket")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Data", len(df))
    with col2:
        st.metric("Jumlah Fitur", len(df.columns) - 1)
    with col3:
        st.metric("Target", "Durasi (jam)")
    with col4:
        avg_durasi = df['Durasi'].mean()
        st.metric("Rata-rata Durasi", f"{avg_durasi:.1f} jam")
    
    st.subheader("Preview Data")
    st.dataframe(df.head(15), use_container_width=True)
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualisasi distribusi
    st.subheader("Distribusi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribusi Jarak vs Durasi
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['Jarak'], df['Durasi'], alpha=0.6, color='steelblue')
        ax.set_xlabel('Jarak (km)')
        ax.set_ylabel('Durasi (jam)')
        ax.set_title('Korelasi Jarak vs Durasi')
        correlation = df['Jarak'].corr(df['Durasi'])
        ax.text(0.05, 0.95, f'Korelasi: {correlation:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig)
    
    with col2:
        # Distribusi per Layanan
        fig, ax = plt.subplots(figsize=(8, 5))
        layanan_avg = df.groupby('Layanan')['Durasi'].mean().sort_values()
        layanan_avg.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Rata-rata Durasi (jam)')
        ax.set_ylabel('Layanan')
        ax.set_title('Rata-rata Durasi per Layanan')
        st.pyplot(fig)
    
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tipe Data:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ Tidak ada missing values")
        else:
            st.write(missing)

# Tab 2: Model Training
with tab2:
    st.header("Training Model Decision Tree")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Preprocessing
            df_encoded = df.copy()
            label_encoders = {}
            
            # Encode categorical features
            categorical_cols = ['Layanan', 'Cuaca']
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
        
        # Interpretasi hasil
        st.markdown("---")
        st.subheader("📋 Interpretasi Hasil")
        
        if r2_test > 0.8:
            st.success(f"✅ Model sangat baik! R² = {r2_test:.4f} (>0.8)")
        elif r2_test > 0.6:
            st.info(f"ℹ️ Model cukup baik. R² = {r2_test:.4f} (0.6-0.8)")
        else:
            st.warning(f"⚠️ Model perlu improvement. R² = {r2_test:.4f} (<0.6)")
        
        st.write(f"**MAE = {mae_test:.2f} jam** → Model rata-rata meleset {mae_test:.2f} jam ({mae_test/24:.2f} hari)")
        
        # Feature Importance
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': st.session_state['feature_names'],
            'Importance': st.session_state['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=feature_imp, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title('Feature Importance - Fitur Mana yang Paling Berpengaruh?')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
        
        with col2:
            st.write("**Ranking Fitur:**")
            for idx, row in feature_imp.iterrows():
                percentage = row['Importance'] * 100
                st.write(f"{row['Feature']}: **{percentage:.1f}%**")

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
                jarak = st.number_input("Jarak (km)", min_value=0, max_value=300, value=50, 
                                       key=f'jarak_{st.session_state["reset_counter"]}',
                                       help="Jarak pengiriman dalam kilometer")
                berat = st.number_input("Berat (kg)", min_value=0, max_value=100, value=5,
                                       key=f'berat_{st.session_state["reset_counter"]}',
                                       help="Berat paket dalam kilogram")
            
            with col2:
                layanan = st.selectbox("Layanan", df['Layanan'].unique(),
                                      key=f'layanan_{st.session_state["reset_counter"]}',
                                      help="Pilih jenis layanan pengiriman")
                cuaca = st.selectbox("Cuaca", df['Cuaca'].unique(),
                                    key=f'cuaca_{st.session_state["reset_counter"]}',
                                    help="Kondisi cuaca saat pengiriman")
            
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
            
            # Create input array
            input_data = np.array([[jarak, berat, layanan_encoded, cuaca_encoded]])
            
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
                'durasi_prediksi': round(prediction, 1)
            }
            st.session_state['prediction_history'].insert(0, prediction_record)
            
            # Keep only last 10 predictions
            if len(st.session_state['prediction_history']) > 10:
                st.session_state['prediction_history'] = st.session_state['prediction_history'][:10]
            
            # Display result
            st.markdown("---")
            st.subheader("✨ Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durasi Prediksi", f"{prediction:.1f} jam", 
                         help="Estimasi waktu pengiriman dalam jam")
            with col2:
                st.metric("Estimasi Hari", f"{prediction/24:.1f} hari",
                         help="Estimasi waktu pengiriman dalam hari")
            with col3:
                if prediction < 24:
                    status = "🟢 Cepat"
                    status_color = "green"
                elif prediction < 72:
                    status = "🟡 Sedang"
                    status_color = "orange"
                else:
                    status = "🔴 Lambat"
                    status_color = "red"
                st.metric("Status", status)
            
            # Detail prediksi dengan styling
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {status_color};'>
                <h4 style='color: #333; margin-top: 0;'>📦 Detail Pengiriman:</h4>
                <ul style='color: #333; font-size: 1rem;'>
                    <li><b>Jarak:</b> {jarak} km</li>
                    <li><b>Berat:</b> {berat} kg</li>
                    <li><b>Layanan:</b> {layanan}</li>
                    <li><b>Cuaca:</b> {cuaca}</li>
                    <li><b>Estimasi Durasi:</b> <span style='color: {status_color}; font-size: 1.2em; font-weight: bold;'>{prediction:.1f} jam ({prediction/24:.1f} hari)</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
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
                elif durasi < 72:
                    return "🟡 Sedang"
                else:
                    return "🔴 Lambat"
            
            history_df['status'] = history_df['durasi_prediksi'].apply(get_status)
            
            # Reorder columns
            display_df = history_df[['timestamp', 'jarak', 'berat', 'layanan', 'cuaca', 
                                    'durasi_prediksi', 'durasi_hari', 'status']]
            
            # Rename columns for better display
            display_df.columns = ['Waktu', 'Jarak (km)', 'Berat (kg)', 'Layanan', 
                                 'Cuaca', 'Durasi (jam)', 'Durasi (hari)', 'Status']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# Tab 4: Visualisasi
with tab4:
    st.header("Visualisasi Decision Tree")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Model Training'")
    else:
        st.subheader("🌳 Struktur Decision Tree")
        
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
        ax.set_title('Struktur Decision Tree - Alur Keputusan Model', fontsize=16, pad=20)
        st.pyplot(fig)
        
        st.info("💡 **Cara Membaca Tree:** Setiap kotak berisi kondisi (misal: Jarak <= 50), nilai sampel di node tersebut, dan prediksi durasi.")
        
        st.markdown("---")
        
        # Actual vs Predicted
        st.subheader("📊 Perbandingan Nilai Aktual vs Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Set")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state['y_train'], st.session_state['y_pred_train'], alpha=0.5)
            ax.plot([st.session_state['y_train'].min(), st.session_state['y_train'].max()], 
                    [st.session_state['y_train'].min(), st.session_state['y_train'].max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Durasi Aktual (jam)')
            ax.set_ylabel('Durasi Prediksi (jam)')
            ax.set_title('Training Set: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Testing Set")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state['y_test'], st.session_state['y_pred_test'], alpha=0.5, color='green')
            ax.plot([st.session_state['y_test'].min(), st.session_state['y_test'].max()], 
                    [st.session_state['y_test'].min(), st.session_state['y_test'].max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Durasi Aktual (jam)')
            ax.set_ylabel('Durasi Prediksi (jam)')
            ax.set_title('Testing Set: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.info("💡 **Interpretasi:** Semakin dekat titik-titik ke garis merah, semakin akurat prediksi model.")
        
        # Residual plot
        st.markdown("---")
        st.subheader("📉 Residual Plot")
        residuals = st.session_state['y_test'] - st.session_state['y_pred_test']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(st.session_state['y_pred_test'], residuals, alpha=0.5, color='purple')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Nilai Prediksi (jam)')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residual Plot - Analisis Error Model')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.info("💡 **Interpretasi:** Residual yang tersebar acak di sekitar garis 0 menunjukkan model bagus. Jika ada pola, model perlu ditingkatkan.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📦 Aplikasi Prediksi Durasi Pengiriman Paket | Tugas UAS Machine Learning</p>
        <p style='font-size: 0.9em;'>Dataset: 4 Fitur (Jarak, Berat, Layanan, Cuaca) → Durasi</p>
    </div>
""", unsafe_allow_html=True)
