import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Aktivitas MS Teams UIN Jakarta",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema warna kustom untuk visualisasi
CUSTOM_COLORS = ['#1E88E5', '#FFC107', '#004D40', '#D81B60']

# ==========================================
# FUNGSI CACHE UNTUK MEMPROSES DATA
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    file_path = 'dataset_teams_uinjkt.xlsx'
    
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_excel(file_path)
    
    # 1. Menangani Missing Value
    df['Role'] = df['Role'].fillna(df['Role'].mode()[0])
    df = df.dropna(subset=['Last Activity Date'])
    
    # 2. Standardisasi Data
    df['Role'] = df['Role'].replace({'tendik': 'Tendik'})
    
    # 3. PENGELOLAAN PRIVASI (Username dipertahankan untuk Papan Peringkat)
    kolom_privasi = ['Email', 'Name', 'Display Name', 'User Principal Name']
    df = df.drop(columns=[col for col in kolom_privasi if col in df.columns], errors='ignore')
    
    # Membuat ID Pengguna Anonim untuk tampilan publik (misal: User_0001)
    df.insert(0, 'User ID', ['User_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
    
    # Memastikan kolom Username ada (sebagai identifier asli)
    if 'Username' not in df.columns:
        df['Username'] = df['User ID']
    
    # 4. Konversi Detik ke Menit & Feature Engineering
    df['Audio Duration (Menit)'] = df['Audio Duration In Seconds'] / 60
    df['Video Duration (Menit)'] = df['Video Duration In Seconds'] / 60
    df['Screen Share (Menit)'] = df['Screen Share Duration In Seconds'] / 60
    
    df['Total_Duration (Menit)'] = (
        df['Audio Duration (Menit)'] + 
        df['Video Duration (Menit)'] + 
        df['Screen Share (Menit)']
    )
    
    # 5. Membuat Label Target berdasarkan Meeting Count
    def kategori_aktivitas(x):
        if x <= 5: return "Rendah"
        elif x <= 15: return "Sedang"
        else: return "Tinggi"
        
    df['Activity_Level'] = df['Meeting Count'].apply(kategori_aktivitas)
    
    return df

@st.cache_resource
def train_models(df):
    label_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    inverse_label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    
    y = df['Activity_Level'].map(label_mapping)
    fitur = [
        'Audio Duration (Menit)',
        'Video Duration (Menit)',
        'Screen Share (Menit)',
        'Total_Duration (Menit)'
    ]
    X = df[fitur]
    
    # Split Data & Feature Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Models
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    # Evaluasi
    acc_dt = accuracy_score(y_test, dt.predict(X_test_scaled))
    acc_rf = accuracy_score(y_test, rf.predict(X_test_scaled))
    acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
    
    eval_dict = {
        'Model': ['Decision Tree', 'Random Forest', 'KNN'],
        'Accuracy': [acc_dt, acc_rf, acc_knn],
        'Objects': [dt, rf, knn]
    }
    
    return scaler, X_test_scaled, y_test, eval_dict, inverse_label_mapping, rf, fitur

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Microsoft_Teams_14_logo.svg/512px-Microsoft_Teams_14_logo.svg.png", width=80)
st.sidebar.title("MS Teams Analytics")
st.sidebar.markdown("Dashboard Aktivitas MS Teams UIN Jakarta")
st.sidebar.markdown("---")
st.sidebar.success("🔒 **Data Privacy Hybrid**\nIdentitas disamarkan pada ringkasan umum, namun Papan Peringkat menampilkan otoritas nama asli untuk evaluasi institusi.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tim Peneliti (2026):**\n- **Abdullah, S.Kom.**\n- **Syariffah Alvi Tara Udini**")
st.sidebar.markdown("**Instansi:** UIN Syarif Hidayatullah Jakarta")

# ==========================================
# PEMBACAAN DATA UTAMA
# ==========================================
df = load_and_preprocess_data()

if df is None:
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.error("⚠️ File `dataset_teams_uinjkt.xlsx` tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan file `app.py` ini.")
else:
    scaler, X_test_scaled, y_test, eval_dict, inv_map, model_terpilih, fitur_names = train_models(df)
    
    # JUDUL UTAMA (Direvisi dengan Periode 3 Bulan)
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.markdown("### Periode 3 Bulan (Maret-Mei 2026) | UIN Syarif Hidayatullah Jakarta")
    st.markdown("---")
    
    # Membuat Tab Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Beranda & Ringkasan", 
        "📈 Exploratory Data Analysis (EDA)", 
        "🤖 Evaluasi Model", 
        "🚀 Simulasi Prediksi (1 Bulan)"
    ])
    
    # ----------------------------------------
    # TAB 1: BERANDA & RINGKASAN
    # ----------------------------------------
    with tab1:
        st.markdown("""
        <div style="background-color:#004D40;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📝 MENU BERANDA & RINGKASAN EKSEKUTIF</h2>
            <p style="color:#E0F2F1;margin:5px 0 0 0">Pusat Informasi Utama dan Statistik Dasar Penggunaan Microsoft Teams</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        ℹ️ **Panduan Informasi Menu:** Halaman ini menampilkan gambaran besar (*overview*) data aktivitas selama kuartal Maret-Mei 2026. 
        Sistem menjamin kerahasiaan identitas pengguna dengan mengganti email/nama menjadi `User ID` anonim. 
        """)
        
        st.markdown("### 📊 Ringkasan Indikator Kunci (Keseluruhan Data)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel Pengguna", f"{len(df)} Orang")
        col2.metric("Rata-rata Durasi Audio", f"{df['Audio Duration (Menit)'].mean():.1f} Menit")
        col3.metric("Rata-rata Durasi Video", f"{df['Video Duration (Menit)'].mean():.1f} Menit")
        col4.metric("Rata-rata Screen Share", f"{df['Screen Share (Menit)'].mean():.1f} Menit")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Preview Dataset Utama (Telah Dianonimkan)")
        
        kolom_tampil = ['User ID', 'Role', 'Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)', 'Total_Duration (Menit)', 'Activity_Level']
        st.dataframe(df[kolom_tampil].head(15), use_container_width=True, hide_index=True)
        st.caption("Catatan: Identitas asli pengguna telah dihapus pada preview ini. Data dikonversi ke satuan Menit.")
        
    # ----------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ----------------------------------------
    with tab2:
