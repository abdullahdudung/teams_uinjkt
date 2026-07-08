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
    df_mhs['Role'] = 'Mahasiswa' # Menambahkan pelabelan Role untuk filter gabungan
    df_mhs = proses_metrik_dan_resensi(df_mhs)
    df_mhs.insert(0, 'User ID', ['MHS_' + str(i).zfill(5) for i in range(1, len(df_mhs) + 1)])
    
    # Proses Staff & Pemisahan Dosen dan Tendik
    df_staff['Role'] = df_staff['Role'].fillna('Tendik').replace({'tendik': 'Tendik', 'dosen': 'Dosen'})
    df_staff = proses_metrik_dan_resensi(df_staff)
    
    df_dosen = df_staff[df_staff['Role'] == 'Dosen'].copy()
    df_dosen.insert(0, 'User ID', ['DSN_' + str(i).zfill(4) for i in range(1, len(df_dosen) + 1)])
    
    df_tendik = df_staff[df_staff['Role'] == 'Tendik'].copy()
    df_tendik.insert(0, 'User ID', ['TDK_' + str(i).zfill(4) for i in range(1, len(df_tendik) + 1)])
    
    # Gabungkan semua data untuk EDA Terpadu & Training Model Machine Learning
    df_all = pd.concat([df_mhs, df_dosen, df_tendik], ignore_index=True)
    
    return df_mhs, df_dosen, df_tendik, df_all

@st.cache_resource
def train_models(df):
    label_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    inverse_label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    
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
    
    # Navigasi Empat Tab Utama (DISEDERHANAKAN)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Ringkasan", 
        "📈 Exploratory Data Analysis (EDA Terpadu)", 
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
    # TAB 2: EXPLORATORY DATA ANALYSIS (TERPADU)
    # ----------------------------------------
    with tab2:
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📈 EXPLORATORY DATA ANALYSIS (EDA TERPADU)</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Pusat Analisis Visual Perilaku Digital, Korelasi, dan Kinerja Seluruh Sivitas Akademika</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. FILTERING DINAMIS (Utama & Sub-Kategori)
        st.markdown("### 🔍 Filter Data Global")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pilihan_role = st.selectbox("Pilih Peran (*Role*) Sivitas:", ["Semua Role", "Mahasiswa", "Dosen", "Tendik"])
            
        df_eda = df_all.copy()
        pilihan_sub = "Semua" # Default value
        
        if pilihan_role != "Semua Role":
            df_eda = df_eda[df_eda['Role'] == pilihan_role].copy()
            
            with col_f2:
                if pilihan_role == "Mahasiswa":
                    list_sub = ["Semua Fakultas"] + sorted(df_eda['Fakultas'].dropna().unique().tolist())
                    pilihan_sub = st.selectbox("Pilih Fakultas:", list_sub)
                    if pilihan_sub != "Semua Fakultas":
                        df_eda = df_eda[df_eda['Fakultas'] == pilihan_sub]
                else:
                    list_sub = ["Semua Unit Kerja"] + sorted(df_eda['Unit Kerja'].dropna().unique().tolist())
                    pilihan_sub = st.selectbox(f"Pilih Unit Kerja ({pilihan_role}):", list_sub)
                    if pilihan_sub != "Semua Unit Kerja":
                        df_eda = df_eda[df_eda['Unit Kerja'] == pilihan_sub]
        else:
            with col_f2:
                st.info("Pilih Peran spesifik (Mahasiswa/Dosen/Tendik) di sebelah kiri untuk memunculkan filter lanjutan (Fakultas/Unit Kerja).")

        st.markdown("---")
        
        # 2. STATUS RESENSI & RATA-RATA DURASI
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            df_status = df_eda['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
            df_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah']
            fig_rec = px.bar(df_status, x='Tingkat Aktivitas Resensi', y='Jumlah', text='Jumlah', 
                             title=f"Distribusi Status Akses (Resensi) - {pilihan_role}", 
                             color='Tingkat Aktivitas Resensi', 
                             color_discrete_sequence=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'])
            fig_rec.update_layout(showlegend=False)
            st.plotly_chart(fig_rec, use_container_width=True)
            
        with col_v2:
            avg_df = pd.DataFrame({
                'Fitur': ['Audio', 'Video', 'Screen Share'],
                'Rata-rata (Jam)': [df_eda['Audio Duration (Jam)'].mean(), df_eda['Video Duration (Jam)'].mean(), df_eda['Screen Share (Jam)'].mean()]
            })
            fig_avg = px.bar(avg_df, x='Fitur', y='Rata-rata (Jam)', text='Rata-rata (Jam)', 
                             title=f"Rata-rata Durasi Fitur - {pilihan_role}", 
                             color='Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'])
            fig_avg.update_traces(texttemplate='%{text:.2f} Jam')
            fig_avg.update_layout(showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)
            
        st.markdown("---")
        
        # 3. PAPAN PERINGKAT TOP 10 INDIVIDU (Bereaksi terhadap Filter)
        st.markdown(f"### 🏆 Top 10 Pengguna Teraktif ({pilihan_role} - {pilihan_sub})")
        t_i1, t_i2, t_i3, t_i4 = st.tabs(["📊 Frekuensi Rapat/Kelas", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        
        if len(df_eda) > 0:
            with t_i1:
                st.plotly_chart(px.bar(df_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', color='Role', orientation='h', text_auto='.0f', title='Top 10: Frekuensi Pertemuan (Meeting Count)'), use_container_width=True)
            with t_i2:
                st.plotly_chart(px.bar(df_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', color='Role', orientation='h', text_auto='.1f', title='Top 10: Durasi Audio (Jam)'), use_container_width=True)
            with t_i3:
                st.plotly_chart(px.bar(df_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', color='Role', orientation='h', text_auto='.1f', title='Top 10: Durasi Video (Jam)'), use_container_width=True)
            with t_i4:
                st.plotly_chart(px.bar(df_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', color='Role', orientation='h', text_auto='.1f', title='Top 10: Durasi Screen Share (Jam)'), use_container_width=True)
        else:
            st.warning("Tidak ada data yang tersedia untuk parameter filter yang dipilih.")

        st.markdown("---")
        
        # 4. RENDER ANALISIS LANJUTAN (Korelasi, Boxplot, 3D Scatter)
        if len(df_eda) > 0:
            render_eda_lanjutan(df_eda, f"Kategori: {pilihan_role}")

    # ----------------------------------------
    # TAB 3: EVALUASI MODEL
    # ----------------------------------------
    with tab3:
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
    # TAB 4: SIMULASI PREDIKSI
    # ----------------------------------------
    with tab4:
        st.markdown("""
        <div style="background-color:#D81B60;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🚀 SIMULASI PREDIKSI AI (AKTIVITAS BULANAN)</h2>
            <p style="color:#FCE4EC;margin:5px 0 0 0">Sistem Kecerdasan Buatan untuk Mengukur & Memprediksi Performa Kolaborasi Digital (Skala Jam)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Cara Kerja AI:** Masukkan estimasi beban jam kerja Anda selama sebulan. Sistem akan mengklasifikasikan kebiasaan Anda berdasarkan database ribuan sivitas akademika lainnya.")
        
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
