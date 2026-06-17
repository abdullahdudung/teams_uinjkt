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
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📈 MENU EXPLORATORY DATA ANALYSIS (EDA)</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Eksplorasi Karakteristik, Korelasi, dan Ranking Individu Pengguna secara Interaktif</p>
        </div>
        """, unsafe_allow_html=True)
        
        # FILTERING DATA
        st.markdown("### 🔍 Filter Data Visualisasi")
        col_filt1, col_filt2 = st.columns([1, 2])
        with col_filt1:
            pilihan_role = st.selectbox("Pilih Peran (*Role*) untuk dianalisis:", ["Semua Data", "Dosen", "Tendik"])
        with col_filt2:
            st.info(f"Visualisasi di bawah ini menampilkan data untuk kategori: **{pilihan_role}**")
            
        if pilihan_role == "Semua Data":
            df_eda = df.copy()
        else:
            df_eda = df[df['Role'] == pilihan_role].copy()
            
        st.markdown("---")
        
        # 1. KOMPOSISI DAN KELAS
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if pilihan_role == "Semua Data":
                role_counts = df_eda['Role'].value_counts().reset_index()
                role_counts.columns = ['Role', 'Jumlah']
                fig_pie = px.pie(role_counts, values='Jumlah', names='Role', hole=0.5, 
                                 color_discrete_sequence=CUSTOM_COLORS, title="Komposisi Pengguna")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                with st.expander("💡 Insight Penggunaan MS Teams: Komposisi Dosen vs Tendik", expanded=True):
                    st.markdown("""
                    **Interpretasi Operasional:** Proporsi peran pada grafik ini memberikan petunjuk tentang arah utama adopsi Microsoft Teams di kampus. 
                    * Jika **Dosen** mendominasi, artinya Teams diandalkan sebagai alat *e-learning* (perkuliahan sinkron daring, bimbingan skripsi). 
                    * Jika **Tendik** mendominasi, artinya platform ini telah menjadi tulang punggung koordinasi administrasi internal (rapat fakultas, rektorat).
                    """)
            else:
                st.metric(f"Total Pengguna ({pilihan_role})", f"{len(df_eda)} Orang")
                st.markdown(f"*(Visualisasi proporsi peran disembunyikan karena Anda memfilter khusus **{pilihan_role}**)*")
                
        with col_eda2:
            act_counts = df_eda['Activity_Level'].value_counts().reset_index()
            act_counts.columns = ['Activity_Level', 'Jumlah']
            act_counts['Activity_Level'] = pd.Categorical(act_counts['Activity_Level'], categories=["Rendah", "Sedang", "Tinggi"], ordered=True)
            act_counts = act_counts.sort_values('Activity_Level')
            
            fig_bar = px.bar(act_counts, x='Activity_Level', y='Jumlah', text='Jumlah', 
                             color='Activity_Level', 
                             color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
                             title=f"Distribusi Tingkat Aktivitas ({pilihan_role})")
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            with st.expander("💡 Insight Penggunaan MS Teams: Tingkat Aktivitas", expanded=True):
                st.markdown("""
                **Interpretasi Operasional:**
                * **Banyak di Kelas 'Rendah':** Menunjukkan penggunaan Teams mungkin masih sporadis.
                * **Banyak di Kelas 'Tinggi' & 'Sedang':** Menandakan bahwa budaya kerja *Hybrid* di UIN Jakarta berjalan dengan baik. Platform tidak lagi dianggap sekadar alternatif, melainkan telah menjadi ruang kerja utama.
                """)

        st.markdown("---")
        
        # 2. BAR CHART: PERBANDINGAN RATA-RATA DURASI FITUR
        st.markdown("### 📊 Perbandingan Rata-rata Durasi Penggunaan Fitur")
        
        avg_data = {
            'Jenis Fitur': ['Audio (Suara)', 'Video (Kamera)', 'Screen Share (Presentasi)'],
            'Rata-rata Durasi (Menit)': [
                df_eda['Audio Duration (Menit)'].mean(),
                df_eda['Video Duration (Menit)'].mean(),
                df_eda['Screen Share (Menit)'].mean()
            ]
        }
        df_avg = pd.DataFrame(avg_data)
        
        fig_avg_bar = px.bar(
            df_avg, x='Jenis Fitur', y='Rata-rata Durasi (Menit)', text='Rata-rata Durasi (Menit)',
            color='Jenis Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'],
            title=f"Rata-rata Durasi Fitur per Pengguna ({pilihan_role})"
        )
        fig_avg_bar.update_traces(texttemplate='%{text:.1f} Menit', textposition='outside')
        fig_avg_bar.update_layout(showlegend=False, yaxis_range=[0, df_avg['Rata-rata Durasi (Menit)'].max() * 1.2])
        st.plotly_chart(fig_avg_bar, use_container_width=True)

        with st.expander("💡 Insight Penggunaan MS Teams: Preferensi Fitur Kolaborasi", expanded=True):
            st.markdown("""
            **Interpretasi Operasional:**
            Grafik batang ini menyoroti **fitur mana yang paling dominan** digunakan oleh civitas akademika UIN Jakarta selama 3 bulan terakhir.
            * **Dominasi Audio:** Tingginya penggunaan *Audio* dibandingkan *Video* dan *Screen Share* mengindikasikan bahwa banyak sesi rapat/perkuliahan berjalan secara asimetris (Dosen atau pimpinan rapat berbicara, sementara sebagian besar partisipan hanya mendengarkan).
            * **Keseimbangan Video & Screen Share:** Jika rata-rata durasi *Video* dan *Screen Share* mulai menyusul atau tidak tertinggal terlalu jauh, ini adalah sinyal positif. Hal ini menunjukkan tingkat kolaborasi dua arah yang interaktif, di mana pemaparan materi (Screen Share) senantiasa diiringi dengan kehadiran tatap muka virtual (Video).
            """)

        st.markdown("---")

        # 3. STATISTIK RINGKASAN & HEATMAP KORELASI
        st.markdown("### 🔗 Analisis Korelasi & Statistik Ringkasan")
        
        col_stat1, col_stat2 = st.columns([1, 1.5])
        
        with col_stat1:
            st.markdown("**Statistik Deskriptif Numerik**")
            st.dataframe(
                df_eda[['Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)']].describe().round(2), 
                use_container_width=True
            )
            st.caption("Menampilkan metrik statistik (Mean, Min, Max, Kuartil) untuk mengukur sebaran data.")
            
        with col_stat2:
            st.markdown("**Heatmap Korelasi Antar Fitur Aktivitas**")
            corr_matrix = df_eda[['Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)', 'Total_Duration (Menit)']].corr()
            
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale='Blues',
                labels=dict(color="Korelasi")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            with st.expander("💡 Insight Penggunaan MS Teams: Heatmap Korelasi", expanded=True):
                st.markdown("""
                **Interpretasi Operasional:**
                Nilai korelasi mendekati **1.00** menandakan hubungan linear positif yang sangat kuat. 
                * Jika korelasi antara *Audio* dan *Video* tinggi (contoh > 0.70), artinya pengguna yang sering berbicara dalam rapat juga memiliki budaya kedisiplinan yang baik untuk menyalakan kamera.
                * Jika nilai *Meeting Count* berkorelasi lemah dengan durasi fitur lainnya, ini menandakan banyak pengguna yang masuk ke dalam ruang rapat virtual, namun bertindak pasif (hanya *join*).
                """)

        st.markdown("---")

        # 4. PAPAN PERINGKAT PENGGUNA
        st.markdown("### 🏆 Papan Peringkat Kinerja Individu (Top 10 Pengguna)")
        st.success("🔓 **Otorisasi Publikasi Data:** Informasi identitas asli (Username/Email) pada panel klasemen ini telah diotorisasi untuk ditampilkan guna mengevaluasi kinerja dan apresiasi institusi.")

        tab_meet, tab_aud, tab_vid, tab_scr = st.tabs([
            "📊 Frekuensi Rapat", 
            "🎙️ Durasi Audio", 
            "📹 Durasi Video", 
            "💻 Durasi Screen Share"
        ])

        with tab_meet:
            top_meet = df_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count', ascending=True)
            fig_meet = px.bar(top_meet, x='Meeting Count', y='Username', color='Role', orientation='h', 
                              text='Meeting Count', color_discrete_map={"Dosen": "#1E88E5", "Tendik": "#FFC107"},
                              title=f"Top 10: Frekuensi Rapat Terbanyak ({pilihan_role})")
            st.plotly_chart(fig_meet, use_container_width=True)

        with tab_aud:
            top_aud = df_eda.nlargest(10, 'Audio Duration (Menit)').sort_values('Audio Duration (Menit)', ascending=True)
            fig_aud = px.bar(top_aud, x='Audio Duration (Menit)', y='Username', color='Role', orientation='h', 
                             text='Audio Duration (Menit)', color_discrete_map={"Dosen": "#1E88E5", "Tendik": "#FFC107"},
                             title=f"Top 10: Komunikasi Suara Terlama ({pilihan_role})")
            fig_aud.update_traces(texttemplate='%{text:.0f}')
            st.plotly_chart(fig_aud, use_container_width=True)

        with tab_vid:
            top_vid = df_eda.nlargest(10, 'Video Duration (Menit)').sort_values('Video Duration (Menit)', ascending=True)
            fig_vid = px.bar(top_vid, x='Video Duration (Menit)', y='Username', color='Role', orientation='h', 
                             text='Video Duration (Menit)', color_discrete_map={"Dosen": "#1E88E5", "Tendik": "#FFC107"},
                             title=f"Top 10: Kamera Menyala Terlama ({pilihan_role})")
            fig_vid.update_traces(texttemplate='%{text:.0f}')
            st.plotly_chart(fig_vid, use_container_width=True)

        with tab_scr:
            top_scr = df_eda.nlargest(10, 'Screen Share (Menit)').sort_values('Screen Share (Menit)', ascending=True)
            fig_scr = px.bar(top_scr, x='Screen Share (Menit)', y='Username', color='Role', orientation='h', 
                             text='Screen Share (Menit)', color_discrete_map={"Dosen": "#1E88E5", "Tendik": "#FFC107"},
                             title=f"Top 10: Presentasi Layar Terlama ({pilihan_role})")
            fig_scr.update_traces(texttemplate='%{text:.0f}')
            st.plotly_chart(fig_scr, use_container_width=True)

        st.markdown("---")
        
        # 5. BOXPLOT DISTRIBUSI DURASI DENGAN INSIGHT BARU
        st.markdown("### 📦 Analisis Kuartil & Deteksi Anomali (Boxplot)")
        
        df_melt = df_eda.melt(id_vars=['User ID', 'Activity_Level'], 
                              value_vars=['Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)'],
                              var_name='Jenis Fitur', value_name='Durasi (Menit)')
                              
        fig_box = px.box(df_melt, x='Jenis Fitur', y='Durasi (Menit)', color='Activity_Level',
                         color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
                         title=f"Sebaran Durasi Fitur berdasarkan Tingkat Aktivitas ({pilihan_role})")
        st.plotly_chart(fig_box, use_container_width=True)
        
        with st.expander("💡 Insight Penggunaan MS Teams: Analisis Kuartil & Anomali (Outliers)", expanded=True):
            st.markdown("""
            **Interpretasi Operasional:**
            Grafik *Boxplot* ini membedah distribusi durasi secara statistik untuk melihat rentang kewajaran aktivitas selama 3 bulan.
            * **Kotak Utama (Kuartil 1 hingga Kuartil 3):** Kotak berwarna menunjukkan rentang waktu normal di mana mayoritas (50% tengah) pengguna beraktivitas. Kotak yang sempit berarti durasi penggunaan antar individu sangat seragam, sementara kotak yang lebar menandakan variasi yang tinggi.
            * **Titik-titik Anomali (Outliers):** Titik-titik individual yang terlempar jauh ke atas melebihi 'kumis' grafik (*whiskers*) **bukanlah data error**. Ini adalah representasi dari **Power Users**. Mereka merupakan individu dengan beban koordinasi digital yang ekstrem (misalnya staf PMB, rektorat, pimpinan fakultas, atau dosen yang memegang banyak sks pengganti).
            """)
            
        st.markdown("---")
        
        # 6. SCATTER PLOT 3D
        st.markdown("### 🌐 Korelasi Multivariat (Audio vs Video vs Screen Share)")
        fig_scatter_3d = px.scatter_3d(
            df_eda, x='Audio Duration (Menit)', y='Video Duration (Menit)', z='Screen Share (Menit)',
            color='Activity_Level', symbol='Activity_Level',
            color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
            opacity=0.7, hover_name='User ID',
            labels={'Activity_Level': 'Tingkat Aktivitas'}
        )
        fig_scatter_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=550)
        st.plotly_chart(fig_scatter_3d, use_container_width=True)
        
        with st.expander("💡 Insight Penggunaan MS Teams: Analisis Sinergi Fitur Lanjutan", expanded=True):
            st.markdown("""
            **Interpretasi Operasional:**
            Grafik 3D ini membuktikan seberapa dalam pengguna mengeksploitasi fitur digital:
            * **Menumpuk di Satu Sumbu (Misal hanya Audio tinggi):** Pengguna mungkin memfungsikan MS Teams layaknya panggilan telepon biasa untuk koordinasi instan, tanpa kebutuhan membedah dokumen kerja bersama.
            * **Klaster Sudut Atas (Tinggi di Ketiga Sumbu):** Ini adalah potret *Interactive Facilitator*. Saat profil ini memimpin sesi, ia tidak sekadar 'hadir'. Ia berbicara aktif (Audio), menatap audiensnya (Video), dan membedah materi kerja secara langsung (Screen Share). Ini adalah standar ideal menuju visi *Smart University*.
            """)
        
    # ----------------------------------------
    # TAB 3: EVALUASI MODEL
    # ----------------------------------------
    with tab3:
        st.markdown("""
        <div style="background-color:#7B1FA2;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🤖 MENU EVALUASI DAN PERFORMA MODEL AI</h2>
            <p style="color:#F3E5F5;margin:5px 0 0 0">Analisis Perbandingan Skor Akurasi Algoritma dan Ekstraksi Parameter Bobot Fitur</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_mod1, col_mod2 = st.columns([1.5, 1])
        with col_mod1:
            df_eval = pd.DataFrame({'Model': eval_dict['Model'], 'Accuracy': eval_dict['Accuracy']})
            df_eval = df_eval.sort_values(by='Accuracy', ascending=True)
            
            fig_acc = px.bar(df_eval, x='Accuracy', y='Model', orientation='h',
                             text=[f"{x:.2%}" for x in df_eval['Accuracy']],
                             color='Model', 
                             color_discrete_sequence=['#FF9800', '#2196F3', '#4CAF50'],
                             title="Tingkat Akurasi Prediksi Berdasarkan Algoritma")
            fig_acc.update_layout(xaxis_range=[0, 1.1], showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col_mod2:
            st.markdown("### 📋 Catatan Teknis Peneliti")
            st.success("""
            **Mengapa Memilih Random Forest untuk Tahap Deployment?**
            Sebagai metode *Ensemble*, algoritma ini bekerja dengan membangun sekumpulan pohon keputusan (100 *decision trees*) lalu mengambil keputusan berbasis mayoritas suara (*majority voting*). Pendekatan ini membuatnya jauh lebih tangguh terhadap *noise* data log.
            """)
            
        st.markdown("---")
        st.markdown("### 🔑 Nilai Kepentingan Fitur (Feature Importance) - Model Random Forest")
        importances = eval_dict['Objects'][1].feature_importances_
        df_imp = pd.DataFrame({'Fitur': fitur_names, 'Bobot Kepentingan': importances}).sort_values(by='Bobot Kepentingan', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Bobot Kepentingan', y='Fitur', orientation='h',
                         title="Atribut yang Paling Berpengaruh dalam Keputusan AI",
                         color_discrete_sequence=['#E91E63'])
        st.plotly_chart(fig_imp, use_container_width=True)

    # ----------------------------------------
    # TAB 4: SIMULASI PREDIKSI (DEPLOYMENT 1 BULAN)
    # ----------------------------------------
    with tab4:
        st.markdown("""
        <div style="background-color:#D81B60;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🚀 SIMULASI PREDIKSI (AKTIVITAS 1 BULAN)</h2>
            <p style="color:#FCE4EC;margin:5px 0 0 0">Prediksi Otomatis Tingkat Aktivitas Bulanan Dosen dan Tendik</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        ℹ️ **Cara Kerja AI:** Sistem Kecerdasan Buatan ini (*Random Forest Classifier*) akan mengelompokkan input Anda dengan cara membandingkan pola durasinya terhadap kebiasaan ribuan pengguna lain di dalam *database* kampus yang sudah dipelajari sebelumnya.
        """)
        
        with st.expander("💡 BUKA UNTUK MELIHAT CONTOH PENGGUNAAN (SKENARIO INPUT)"):
            st.markdown("""
            Jika Anda bingung angka apa yang harus diinput, berikut adalah estimasi cara menghitung durasi aktivitas selama **1 Bulan (4 Minggu)**:
            
            **Skenario A (Contoh Input Dosen Mengajar):**
            * Dosen mengajar 2 Mata Kuliah per minggu via MS Teams. Tiap sesi berlangsung 100 menit. 
            * Maka dalam 1 Bulan (8 sesi):
              * **Durasi Audio:** 800 Menit.
              * **Durasi Video:** 600 Menit.
              * **Durasi Screen Share:** 400 Menit.
              
            **Skenario B (Contoh Input Tendik Rapat):**
            * Tendik melakukan rapat koordinasi mingguan 1 kali seminggu. Tiap sesi 60 Menit.
            * Maka dalam 1 Bulan (4 sesi):
              * **Durasi Audio:** 240 Menit.
              * **Durasi Video:** 120 Menit.
              * **Durasi Screen Share:** 60 Menit.
            """)
        
        with st.form("form_prediksi"):
            st.markdown("#### 📥 Form Input Data Aktivitas Bulanan (30 Hari)")
            col_in1, col_in2, col_in3 = st.columns(3)
            
            with col_in1:
                audio_in = st.number_input("🎙️ Durasi Audio (Menit/Bulan)", min_value=0, value=800, step=50)
            with col_in2:
                video_in = st.number_input("📹 Durasi Video (Menit/Bulan)", min_value=0, value=600, step=50)
            with col_in3:
                screen_in = st.number_input("💻 Durasi Screen Share (Menit/Bulan)", min_value=0, value=400, step=50)
                
            submit_btn = st.form_submit_button("Mulai Analisis AI (Bulanan)", type="primary")
            
        if submit_btn:
            total_in = audio_in + video_in + screen_in
            data_baru = pd.DataFrame({
                'Audio Duration (Menit)': [audio_in],
                'Video Duration (Menit)': [video_in],
                'Screen Share (Menit)': [screen_in],
                'Total_Duration (Menit)': [total_in]
            })
            
            data_baru_scaled = scaler.transform(data_baru)
            pred_kode = model_terpilih.predict(data_baru_scaled)[0]
            hasil_prediksi = inv_map[pred_kode]
            proba = model_terpilih.predict_proba(data_baru_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🔔 Hasil Analisis Performa Bulanan (1 Bulan)")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Total Akumulasi Durasi Bulanan", f"{total_in} Menit")
                if hasil_prediksi == "Rendah":
                    st.markdown("Performa Bulanan: <br><span style='color:#EF5350;font-weight:bold;font-size:28px'>RENDAH</span>", unsafe_allow_html=True)
                elif hasil_prediksi == "Sedang":
                    st.markdown("Performa Bulanan: <br><span style='color:#FFCA28;font-weight:bold;font-size:28px'>SEDANG</span>", unsafe_allow_html=True)
                else:
                    st.markdown("Performa Bulanan: <br><span style='color:#66BB6A;font-weight:bold;font-size:28px'>TINGGI</span>", unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("**📊 Tingkat Keyakinan Keputusan Model AI (*Class Probabilities*):**")
                st.progress(float(proba[0]), text=f"Kemungkinan Masuk Kelas Rendah: {proba[0]:.1%}")
                st.progress(float(proba[1]), text=f"Kemungkinan Masuk Kelas Sedang: {proba[1]:.1%}")
                st.progress(float(proba[2]), text=f"Kemungkinan Masuk Kelas Tinggi: {proba[2]:.1%}")
            
            st.markdown("### 📌 Kesimpulan Sistem & Rekomendasi Institusi:")
            if hasil_prediksi == "Rendah":
                st.error("""
                **Kesimpulan (Kategori RENDAH):** Kalkulasi AI menunjukkan bahwa aktivitas pengguna sangat minim di bulan ini.
                
                **Rekomendasi Tindakan:** Pastikan tidak ada kendala teknis pada akun atau perangkat pengguna. Dorong partisipasi aktif melalui sosialisasi e-learning.
                """)
            elif hasil_prediksi == "Sedang":
                st.warning("""
                **Kesimpulan (Kategori SEDANG):** Performa bulanan berada di level wajar.
                
                **Rekomendasi Tindakan:** Penggunaan sudah cukup memadai. Berikan imbauan ringan agar interaksi visual (kamera) dapat lebih ditingkatkan.
                """)
            else:
                st.success("""
                **Kesimpulan (Kategori TINGGI):** Keterlibatan bulanan sangat impresif (*Power User*).
                
                **Rekomendasi Tindakan:** Jadikan individu ini sebagai percontohan (*Role Model*) tata kelola kolaborasi jarak jauh di fakultas/unit kerjanya.
                """)
