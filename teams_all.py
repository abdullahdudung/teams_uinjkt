import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Aktivitas MS Teams UIN Jakarta",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema warna kustom
CUSTOM_COLORS = ['#1E88E5', '#FFC107', '#004D40', '#D81B60']

# ==========================================
# FUNGSI CACHE UNTUK MEMPROSES DATA
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    mhs_file = 'mhs juni.csv'
    staff_file = 'staff juni.csv'
    
    if not os.path.exists(mhs_file) or not os.path.exists(staff_file):
        return None, None, None, None
        
    df_mhs = pd.read_csv(mhs_file)
    df_staff = pd.read_csv(staff_file)
    
    def proses_metrik_dan_resensi(df):
        # 1. Konversi Durasi dari Detik ke JAM (Hours)
        df['Audio Duration (Jam)'] = df['Audio Duration In Seconds'] / 3600
        df['Video Duration (Jam)'] = df['Video Duration In Seconds'] / 3600
        df['Screen Share (Jam)'] = df['Screen Share Duration In Seconds'] / 3600
        df['Total_Duration (Jam)'] = (
            df['Audio Duration (Jam)'] + 
            df['Video Duration (Jam)'] + 
            df['Screen Share (Jam)']
        )
        
        # 2. Parsing Tanggal untuk Analisis Resensi Aktivitas
        df['Report Refresh Date DT'] = pd.to_datetime(df['Report Refresh Date'], dayfirst=True, errors='coerce')
        df['Last Activity Date DT'] = pd.to_datetime(df['Last Activity Date'], dayfirst=True, errors='coerce')
        df['Hari Sejak Akses Terakhir'] = (df['Report Refresh Date DT'] - df['Last Activity Date DT']).dt.days
        
        # 3. Klasifikasi Tingkat Aktivitas Berdasarkan Akses Terakhir
        def golongkan_resensi(row):
            if pd.isna(row['Last Activity Date DT']):
                return "Tidak Aktif (Dalam 180 Hari)"
            hari = row['Hari Sejak Akses Terakhir']
            if hari <= 7: return "Sangat Aktif (Akses 0-7 Hari Lalu)"
            elif hari <= 30: return "Aktif (Akses 8-30 Hari Lalu)"
            elif hari <= 90: return "Cukup Aktif (Akses 31-90 Hari Lalu)"
            else: return "Pasif (Akses >90 Hari Lalu)"
                
        df['Tingkat_Aktivitas_Recency'] = df.apply(golongkan_resensi, axis=1)
        
        # 4. Target Label untuk Machine Learning (Berdasarkan Frekuensi Rapat)
        def kategori_aktivitas(x):
            if x <= 5: return "Rendah"
            elif x <= 15: return "Sedang"
            else: return "Tinggi"
        df['Activity_Level'] = df['Meeting Count'].apply(kategori_aktivitas)
        
        # Penanganan Nama untuk Papan Peringkat Individu
        if 'Nama' in df.columns:
            df['Nama_Tampil'] = df['Nama'].fillna(df['Username'])
        else:
            df['Nama_Tampil'] = df['Username']
            
        return df

    # Proses Mahasiswa
    df_mhs = proses_metrik_dan_resensi(df_mhs)
    df_mhs.insert(0, 'User ID', ['MHS_' + str(i).zfill(5) for i in range(1, len(df_mhs) + 1)])
    
    # Proses Staff & Pemisahan Dosen dan Tendik
    df_staff['Role'] = df_staff['Role'].fillna('Tendik').replace({'tendik': 'Tendik', 'dosen': 'Dosen'})
    df_staff = proses_metrik_dan_resensi(df_staff)
    
    df_dosen = df_staff[df_staff['Role'] == 'Dosen'].copy()
    df_dosen.insert(0, 'User ID', ['DSN_' + str(i).zfill(4) for i in range(1, len(df_dosen) + 1)])
    
    df_tendik = df_staff[df_staff['Role'] == 'Tendik'].copy()
    df_tendik.insert(0, 'User ID', ['TDK_' + str(i).zfill(4) for i in range(1, len(df_tendik) + 1)])
    
    # Gabungkan semua data untuk training model Machine Learning
    df_all = pd.concat([df_mhs, df_dosen, df_tendik], ignore_index=True)
    
    return df_mhs, df_dosen, df_tendik, df_all

@st.cache_resource
def train_models(df):
    label_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    inverse_label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    
    # Memfilter data yang hanya aktif untuk training akurat
    df_train = df.dropna(subset=['Activity_Level']).copy()
    
    y = df_train['Activity_Level'].map(label_mapping)
    fitur = ['Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)', 'Total_Duration (Jam)']
    X = df_train[fitur]
    
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
    
    return scaler, eval_dict, inverse_label_mapping, rf, fitur

# ==========================================
# PEMBACAAN DATA UTAMA & UI
# ==========================================
df_mhs, df_dosen, df_tendik, df_all = load_and_preprocess_data()

if df_all is None:
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.error("⚠️ File dataset tidak ditemukan di direktori aplikasi.")
else:
    # Training Model secara Global (Cached)
    scaler, eval_dict, inv_map, model_terpilih, fitur_names = train_models(df_all)
    
    # Header Aplikasi Utama
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.markdown("### Laporan & Prediksi Akumulasi 180 Hari | UIN Syarif Hidayatullah Jakarta")
    st.markdown("---")
    
    # Navigasi Enam Tab Utama
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Ringkasan", 
        "🎓 EDA Mahasiswa", 
        "👨‍🏫 EDA Dosen",
        "💼 EDA Tendik",
        "🤖 Evaluasi Model", 
        "🚀 Simulasi Prediksi"
    ])
    
    status_order = ["Sangat Aktif (Akses 0-7 Hari Lalu)", "Aktif (Akses 8-30 Hari Lalu)", "Cukup Aktif (Akses 31-90 Hari Lalu)", "Pasif (Akses >90 Hari Lalu)", "Tidak Aktif (Dalam 180 Hari)"]
    color_map_activity = {"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"}
    
    # ----------------------------------------
    # TAB 1: RINGKASAN INFORMASI
    # ----------------------------------------
    with tab1:
        st.markdown("""
        <div style="background-color:#004D40;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📝 RINGKASAN EKSEKUTIF KESELURUHAN</h2>
            <p style="color:#E0F2F1;margin:5px 0 0 0">Gambaran Makro Adopsi Lisensi Microsoft 365 Kampus Selama Periode 180 Hari</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Lisensi Mahasiswa", f"{len(df_mhs):,}".replace(',', '.'))
        col2.metric("Total Lisensi Dosen", f"{len(df_dosen):,}".replace(',', '.'))
        col3.metric("Total Lisensi Tendik", f"{len(df_tendik):,}".replace(',', '.'))
        
        st.markdown("---")
        
        st.markdown("### 📊 Profil Rata-rata Beban Interaksi Digital (Satuan Jam)")
        col_prof1, col_prof2, col_prof3 = st.columns(3)
        
        with col_prof1:
            st.markdown("#### 🎓 Mahasiswa")
            st.info(f"🎙️ Audio: **{df_mhs['Audio Duration (Jam)'].mean():.2f} Jam**\n\n📹 Video: **{df_mhs['Video Duration (Jam)'].mean():.2f} Jam**\n\n💻 Screen Share: **{df_mhs['Screen Share (Jam)'].mean():.2f} Jam**")
        with col_prof2:
            st.markdown("#### 👨‍🏫 Dosen")
            st.success(f"🎙️ Audio: **{df_dosen['Audio Duration (Jam)'].mean():.2f} Jam**\n\n📹 Video: **{df_dosen['Video Duration (Jam)'].mean():.2f} Jam**\n\n💻 Screen Share: **{df_dosen['Screen Share (Jam)'].mean():.2f} Jam**")
        with col_prof3:
            st.markdown("#### 💼 Tendik")
            st.warning(f"🎙️ Audio: **{df_tendik['Audio Duration (Jam)'].mean():.2f} Jam**\n\n📹 Video: **{df_tendik['Video Duration (Jam)'].mean():.2f} Jam**\n\n💻 Screen Share: **{df_tendik['Screen Share (Jam)'].mean():.2f} Jam**")

    # =========================================================================
    # FUNGSI HELPER UNTUK RENDER EDA KHUSUS (Korelasi, Boxplot, Scatter)
    # =========================================================================
    def render_eda_lanjutan(df_eda, role_name):
        st.markdown(f"### 🔗 Analisis Korelasi & Statistik Ringkasan ({role_name})")
        col_stat1, col_stat2 = st.columns([1, 1.5])
        
        with col_stat1:
            st.markdown("**Statistik Deskriptif (Jam)**")
            st.dataframe(df_eda[['Meeting Count', 'Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)']].describe().round(2), use_container_width=True)
            
        with col_stat2:
            st.markdown("**Heatmap Korelasi Antar Fitur**")
            corr_matrix = df_eda[['Meeting Count', 'Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)', 'Total_Duration (Jam)']].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='Blues')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("---")
        st.markdown(f"### 📦 Analisis Kuartil & Deteksi Anomali (Boxplot - {role_name})")
        df_melt = df_eda.melt(id_vars=['User ID', 'Activity_Level'], 
                              value_vars=['Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)'],
                              var_name='Jenis Fitur', value_name='Durasi (Jam)')
                              
        fig_box = px.box(df_melt, x='Jenis Fitur', y='Durasi (Jam)', color='Activity_Level',
                         color_discrete_map=color_map_activity,
                         title=f"Sebaran Durasi Fitur berdasarkan Tingkat Aktivitas")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("**Interpretasi:** Titik di luar kumis (outliers) menunjukkan individu (Power Users) yang jam terbangnya ekstrem dibanding mayoritas koleganya.")
        
        st.markdown("---")
        st.markdown(f"### 🌐 Korelasi Multivariat 3D ({role_name})")
        fig_scatter_3d = px.scatter_3d(
            df_eda, x='Audio Duration (Jam)', y='Video Duration (Jam)', z='Screen Share (Jam)',
            color='Activity_Level', symbol='Activity_Level',
            color_discrete_map=color_map_activity, opacity=0.7, hover_name='Nama_Tampil'
        )
        fig_scatter_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=550)
        st.plotly_chart(fig_scatter_3d, use_container_width=True)

    # ----------------------------------------
    # TAB 2: EDA MAHASISWA
    # ----------------------------------------
    with tab2:
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🎓 EXPLORATORY DATA ANALYSIS: MAHASISWA</h2>
        </div>
        """, unsafe_allow_html=True)
        
        list_fakultas = ["Semua Fakultas"] + sorted(df_mhs['Fakultas'].dropna().unique().tolist())
        pilihan_fakultas = st.selectbox("🔍 Filter Fakultas (Mahasiswa):", list_fakultas, key="filter_mhs")
        df_mhs_eda = df_mhs.copy() if pilihan_fakultas == "Semua Fakultas" else df_mhs[df_mhs['Fakultas'] == pilihan_fakultas].copy()
        
        st.markdown("---")
        
        # Papan Peringkat
        st.markdown("### 🏆 Top 10 Individu Mahasiswa Teraktif")
        t_i1, t_i2, t_i3, t_i4 = st.tabs(["📊 Frekuensi Rapat", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with t_i1:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Mahasiswa: Frekuensi Kelas'), use_container_width=True)
        with t_i2:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Audio (Jam)'), use_container_width=True)
        with t_i3:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Video (Jam)'), use_container_width=True)
        with t_i4:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Screen Share (Jam)'), use_container_width=True)
            
        st.markdown("---")
        # Render Analisis Lanjutan
        render_eda_lanjutan(df_mhs_eda, "Mahasiswa")

    # ----------------------------------------
    # TAB 3: EDA DOSEN
    # ----------------------------------------
    with tab3:
        st.markdown("""
        <div style="background-color:#FF8F00;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">👨‍🏫 EXPLORATORY DATA ANALYSIS: DOSEN</h2>
        </div>
        """, unsafe_allow_html=True)
        
        list_unit_dosen = ["Semua Unit Kerja"] + sorted(df_dosen['Unit Kerja'].dropna().unique().tolist())
        pilihan_unit_dosen = st.selectbox("🔍 Filter Unit Kerja (Dosen):", list_unit_dosen, key="filter_dosen")
        df_dosen_eda = df_dosen.copy() if pilihan_unit_dosen == "Semua Unit Kerja" else df_dosen[df_dosen['Unit Kerja'] == pilihan_unit_dosen].copy()
        
        st.markdown("---")
        
        st.markdown("### 🏆 Top 10 Individu Dosen Teraktif")
        td_i1, td_i2, td_i3, td_i4 = st.tabs(["📊 Frekuensi Mengajar", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with td_i1:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Dosen: Frekuensi Mengajar'), use_container_width=True)
        with td_i2:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Audio (Jam)'), use_container_width=True)
        with td_i3:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Video (Jam)'), use_container_width=True)
        with td_i4:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Screen Share (Jam)'), use_container_width=True)

        st.markdown("---")
        render_eda_lanjutan(df_dosen_eda, "Dosen")

    # ----------------------------------------
    # TAB 4: EDA TENDIK
    # ----------------------------------------
    with tab4:
        st.markdown("""
        <div style="background-color:#6A1B9A;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">💼 EXPLORATORY DATA ANALYSIS: TENDIK</h2>
        </div>
        """, unsafe_allow_html=True)
        
        list_unit_tendik = ["Semua Unit Kerja"] + sorted(df_tendik['Unit Kerja'].dropna().unique().tolist())
        pilihan_unit_tendik = st.selectbox("🔍 Filter Unit Kerja (Tendik):", list_unit_tendik, key="filter_tendik")
        df_tendik_eda = df_tendik.copy() if pilihan_unit_tendik == "Semua Unit Kerja" else df_tendik[df_tendik['Unit Kerja'] == pilihan_unit_tendik].copy()
        
        st.markdown("---")
        
        st.markdown("### 🏆 Top 10 Individu Tendik Teraktif")
        tt_i1, tt_i2, tt_i3, tt_i4 = st.tabs(["📊 Frekuensi Koordinasi", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with tt_i1:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Tendik: Frekuensi Rapat'), use_container_width=True)
        with tt_i2:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Audio (Jam)'), use_container_width=True)
        with tt_i3:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Video (Jam)'), use_container_width=True)
        with tt_i4:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Screen Share (Jam)'), use_container_width=True)

        st.markdown("---")
        render_eda_lanjutan(df_tendik_eda, "Tendik")

    # ----------------------------------------
    # TAB 5: EVALUASI MODEL
    # ----------------------------------------
    with tab5:
        st.markdown("""
        <div style="background-color:#7B1FA2;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🤖 MENU EVALUASI DAN PERFORMA MODEL AI</h2>
            <p style="color:#F3E5F5;margin:5px 0 0 0">Analisis Perbandingan Skor Akurasi Algoritma untuk Penggolongan Aktivitas (Semua Pengguna)</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_mod1, col_mod2 = st.columns([1.5, 1])
        with col_mod1:
            df_eval = pd.DataFrame({'Model': eval_dict['Model'], 'Accuracy': eval_dict['Accuracy']})
            df_eval = df_eval.sort_values(by='Accuracy', ascending=True)
            
            fig_acc = px.bar(df_eval, x='Accuracy', y='Model', orientation='h',
                             text=[f"{x:.2%}" for x in df_eval['Accuracy']],
                             color='Model', color_discrete_sequence=['#FF9800', '#2196F3', '#4CAF50'],
                             title="Tingkat Akurasi Prediksi Berdasarkan Algoritma")
            fig_acc.update_layout(xaxis_range=[0, 1.1], showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col_mod2:
            st.success("""
            **Catatan Teknis Peneliti:**
            AI ini dilatih menggunakan **keseluruhan data agregat (Mahasiswa + Dosen + Tendik)**. 
            Random Forest Classifier kembali diimplementasikan sebagai inti model (Deployment) berkat kemampuannya meminimalkan *noise* data log jam terbang yang sangat bervariasi.
            """)
            
        st.markdown("---")
        st.markdown("### 🔑 Nilai Kepentingan Fitur (Feature Importance) - Model Random Forest")
        importances = model_terpilih.feature_importances_
        df_imp = pd.DataFrame({'Fitur': fitur_names, 'Bobot Kepentingan': importances}).sort_values(by='Bobot Kepentingan', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Bobot Kepentingan', y='Fitur', orientation='h',
                         title="Atribut yang Paling Berpengaruh dalam Penentuan Aktivitas",
                         color_discrete_sequence=['#E91E63'])
        st.plotly_chart(fig_imp, use_container_width=True)

    # ----------------------------------------
    # TAB 6: SIMULASI PREDIKSI (1 BULAN)
    # ----------------------------------------
    with tab6:
        st.markdown("""
        <div style="background-color:#D81B60;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🚀 SIMULASI PREDIKSI AI (AKTIVITAS BULANAN)</h2>
            <p style="color:#FCE4EC;margin:5px 0 0 0">Sistem Kecerdasan Buatan untuk Mengukur & Memprediksi Performa Kolaborasi Digital (Skala Jam)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Cara Kerja AI:** Masukkan beban jam kerja Anda selama sebulan (misalnya mengajar atau rapat virtual). Sistem akan mengklasifikasikan kebiasaan Anda berdasarkan database ribuan sivitas akademika lainnya.")
        
        with st.form("form_prediksi"):
            st.markdown("#### 📥 Form Input Data Aktivitas (Dalam Satuan JAM)")
            col_in1, col_in2, col_in3 = st.columns(3)
            
            with col_in1:
                audio_in = st.number_input("🎙️ Durasi Audio (Jam/Bulan)", min_value=0.0, value=15.0, step=1.0)
            with col_in2:
                video_in = st.number_input("📹 Durasi Video (Jam/Bulan)", min_value=0.0, value=10.0, step=1.0)
            with col_in3:
                screen_in = st.number_input("💻 Durasi Screen Share (Jam/Bulan)", min_value=0.0, value=5.0, step=1.0)
                
            submit_btn = st.form_submit_button("Mulai Analisis AI (Prediksi)", type="primary")
            
        if submit_btn:
            total_in = audio_in + video_in + screen_in
            data_baru = pd.DataFrame({
                'Audio Duration (Jam)': [audio_in],
                'Video Duration (Jam)': [video_in],
                'Screen Share (Jam)': [screen_in],
                'Total_Duration (Jam)': [total_in]
            })
            
            data_baru_scaled = scaler.transform(data_baru)
            pred_kode = model_terpilih.predict(data_baru_scaled)[0]
            hasil_prediksi = inv_map[pred_kode]
            proba = model_terpilih.predict_proba(data_baru_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🔔 Hasil Prediksi AI (1 Bulan)")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Total Akumulasi Beban", f"{total_in} Jam")
                if hasil_prediksi == "Rendah":
                    st.markdown("Prediksi Aktivitas: <br><span style='color:#EF5350;font-weight:bold;font-size:28px'>RENDAH</span>", unsafe_allow_html=True)
                elif hasil_prediksi == "Sedang":
                    st.markdown("Prediksi Aktivitas: <br><span style='color:#FFCA28;font-weight:bold;font-size:28px'>SEDANG</span>", unsafe_allow_html=True)
                else:
                    st.markdown("Prediksi Aktivitas: <br><span style='color:#66BB6A;font-weight:bold;font-size:28px'>TINGGI</span>", unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("**📊 Tingkat Keyakinan Keputusan Model AI (*Class Probabilities*):**")
                st.progress(float(proba[0]), text=f"Probabilitas Kelas Rendah: {proba[0]:.1%}")
                st.progress(float(proba[1]), text=f"Probabilitas Kelas Sedang: {proba[1]:.1%}")
                st.progress(float(proba[2]), text=f"Probabilitas Kelas Tinggi: {proba[2]:.1%}")
