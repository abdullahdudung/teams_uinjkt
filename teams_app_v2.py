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
    
    # 3. Konversi Detik ke Menit & Feature Engineering
    df['Audio Duration (Menit)'] = df['Audio Duration In Seconds'] / 60
    df['Video Duration (Menit)'] = df['Video Duration In Seconds'] / 60
    df['Screen Share (Menit)'] = df['Screen Share Duration In Seconds'] / 60
    
    df['Total_Duration (Menit)'] = (
        df['Audio Duration (Menit)'] + 
        df['Video Duration (Menit)'] + 
        df['Screen Share (Menit)']
    )
    
    # 4. Membuat Label Target berdasarkan Meeting Count
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
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scaling
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
    
    # Evaluasi Akurasi
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
st.sidebar.markdown("Dashboard cerdas pelacakan platform kolaborasi digital kampus.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tim Peneliti (2026):**\n- **Abdullah, S.Kom.** (2250420007)\n- **Syariffah Alvi Tara Udini** (2250420003)")
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
    
    # Membuat Tab Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Beranda & Ringkasan", 
        "📈 Visualisasi Interaktif", 
        "🤖 Evaluasi Model", 
        "🚀 Prediksi Pengguna Baru"
    ])
    
    # ----------------------------------------
    # TAB 1: BERANDA & RINGKASAN
    # ----------------------------------------
    with tab1:
        # HEADER BANNER
        st.markdown("""
        <div style="background-color:#004D40;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📝 MENU BERANDA & RINGKASAN EKSEKUTIF</h2>
            <p style="color:#E0F2F1;margin:5px 0 0 0">Pusat Informasi Utama dan Statistik Dasar Penggunaan Microsoft Teams di UIN Jakarta</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PANDUAN INFORMASI
        st.warning("""
        ℹ️ **Panduan Informasi Menu:** Halaman ini menampilkan gambaran besar (*overview*) data aktivitas dosen dan tenaga kependidikan (Tendik). 
        Gunakan kartu metrik utama di bawah ini untuk melihat rata-rata durasi penggunaan dalam satuan **Menit**. 
        Tabel di bagian bawah menampilkan 15 baris sampel data pertama yang telah dibersihkan dan siap digunakan untuk analisis lanjutan.
        """)
        
        st.markdown("### 📊 Ringkasan Indikator Kunci (KPI)")
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel Pengguna", f"{len(df)} Orang")
        col2.metric("Rata-rata Durasi Audio", f"{df['Audio Duration (Menit)'].mean():.1f} Menit")
        col3.metric("Rata-rata Durasi Video", f"{df['Video Duration (Menit)'].mean():.1f} Menit")
        col4.metric("Rata-rata Screen Share", f"{df['Screen Share (Menit)'].mean():.1f} Menit")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Preview Dataset Utama (Konversi Waktu ke Menit)")
        
        st.dataframe(
            df[['Username', 'Role', 'Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)', 'Activity_Level']].head(15), 
            use_container_width=True,
            hide_index=True
        )
        st.caption("Catatan: Data di atas telah melalui proses standarisasi penulisan kategori 'Role' dan pembersihan data kosong (*missing value*).")
        
    # ----------------------------------------
    # TAB 2: VISUALISASI INTERAKTIF (EDA)
    # ----------------------------------------
    with tab2:
        # HEADER BANNER
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📈 MENU EXPLORATORY DATA ANALYSIS (EDA)</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Eksplorasi Karakteristik, Komposisi, dan Hubungan Antar-Variabel Aktivitas Kampus secara Visual</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PANDUAN INFORMASI
        st.info("""
        ℹ️ **Panduan Informasi & Interaksi Grafik:** * **Grafik Interaktif:** Anda dapat mengarahkan kursor (*hover*) ke area grafik untuk melihat detail angka.
        * **Grafik 3D Scatter Plot:** Klik tahan dan geser mouse Anda pada grafik 3D di bawah untuk memutar sudut pandang, gunakan scroll untuk *zoom-in/out* guna memetakan sebaran pengguna secara mendalam.
        """)
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            # Donut Chart untuk Role
            role_counts = df['Role'].value_counts().reset_index()
            role_counts.columns = ['Role', 'Jumlah']
            fig_pie = px.pie(role_counts, values='Jumlah', names='Role', hole=0.5, 
                             color_discrete_sequence=CUSTOM_COLORS,
                             title="Komposisi Pengguna: Dosen vs Tendik")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # PENJELASAN VISUALISASI
            st.markdown("""
            **📌 Penjelasan Visualisasi Komposisi:** Diagram cincin (*donut chart*) ini memisahkan proporsi data berdasarkan peran civitas akademika. 
            Melalui grafik ini, manajemen IT dapat mengetahui apakah basis data aktivitas didominasi oleh Dosen (terkait perkuliahan) atau Tenaga Kependidikan (terkait administrasi kantor).
            """)
            
        with col_eda2:
            # Bar Chart untuk Level Aktivitas
            act_counts = df['Activity_Level'].value_counts().reset_index()
            act_counts.columns = ['Activity_Level', 'Jumlah']
            act_counts['Activity_Level'] = pd.Categorical(act_counts['Activity_Level'], categories=["Rendah", "Sedang", "Tinggi"], ordered=True)
            act_counts = act_counts.sort_values('Activity_Level')
            
            fig_bar = px.bar(act_counts, x='Activity_Level', y='Jumlah', text='Jumlah', 
                             color='Activity_Level', 
                             color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
                             title="Distribusi Kelas Target (Tingkat Aktivitas)")
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # PENJELASAN VISUALISASI
            st.markdown("""
            **📌 Penjelasan Visualisasi Tingkat Aktivitas:** Grafik batang ini mengelompokkan pengguna ke dalam 3 spektrum aktivitas berdasarkan frekuensi rapat (*Meeting Count*). 
            Ini membantu mengidentifikasi apakah sebaran kelas bersifat seimbang atau terdapat ketimpangan (*class imbalance*) sebelum pemodelan AI dilakukan.
            """)

        st.markdown("---")
        
        # Scatter Plot 3D
        st.markdown("### 🌐 Korelasi Durasi Rapat Multivariat (Audio vs Video vs Screen Share)")
        fig_scatter_3d = px.scatter_3d(
            df, x='Audio Duration (Menit)', y='Video Duration (Menit)', z='Screen Share (Menit)',
            color='Activity_Level', symbol='Role',
            color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
            opacity=0.7,
            labels={'Activity_Level': 'Tingkat Aktivitas'}
        )
        fig_scatter_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=550)
        st.plotly_chart(fig_scatter_3d, use_container_width=True)
        
        # PENJELASAN VISUALISASI
        st.markdown("""
        **📌 Penjelasan Visualisasi 3D Scatter:** Setiap titik mewakili satu pengguna individu. Klaster warna **Hijau (Tinggi)** cenderung berkumpul di area atas menjauhi titik nol koordinat, membuktikan bahwa pengguna aktif memiliki durasi pengerjaan/partisipasi yang selaras di semua lini (aktif berbicara, aktif menyalakan kamera, dan membagikan layar presentasi). Sementara klaster **Merah (Rendah)** menumpuk di sudut dasar grafik.
        """)
        
    # ----------------------------------------
    # TAB 3: EVALUASI MODEL
    # ----------------------------------------
    with tab3:
        # HEADER BANNER
        st.markdown("""
        <div style="background-color:#7B1FA2;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🤖 MENU EVALUASI DAN PERFORMA MODEL AI</h2>
            <p style="color:#F3E5F5;margin:5px 0 0 0">Analisis Perbandingan Skor Akurasi Algoritma dan Ekstraksi Parameter Bobot Fitur</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PANDUAN INFORMASI
        st.info("""
        ℹ️ **Panduan Informasi Komparasi:** Halaman ini membandingkan hasil pengujian dari 3 algoritma *Machine Learning* yang dilatih menggunakan metode klasifikasi terawasi (*supervised learning*). 
        Data telah dipisahkan dengan rasio **80% Training** dan **20% Testing** dengan menerapkan penskalaan fitur (*Feature Scaling*) setelah proses split guna menghindari kebocoran data (*data leakage*).
        """)
        
        col_mod1, col_mod2 = st.columns([1.5, 1])
        
        with col_mod1:
            # Grafik Perbandingan Akurasi (Horizontal)
            df_eval = pd.DataFrame({'Model': eval_dict['Model'], 'Accuracy': eval_dict['Accuracy']})
            df_eval = df_eval.sort_values(by='Accuracy', ascending=True)
            
            fig_acc = px.bar(df_eval, x='Accuracy', y='Model', orientation='h',
                             text=[f"{x:.2%}" for x in df_eval['Accuracy']],
                             color='Model', 
                             color_discrete_sequence=['#FF9800', '#2196F3', '#4CAF50'],
                             title="Tingkat Akurasi Prediksi Berdasarkan Algoritma")
            fig_acc.update_layout(xaxis_range=[0, 1.1], showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # PENJELASAN VISUALISASI
            st.markdown("""
            **📌 Penjelasan Grafik Akurasi:** Bagan horizontal ini memvisualisasikan ketepatan masing-masing model dalam menebak kelas aktivitas pengguna secara benar pada data pengujian. Semakin panjang batang menuju nilai 100%, semakin kredibel model tersebut.
            """)
            
        with col_mod2:
            st.markdown("### 📋 Catatan Teknis Peneliti")
            st.success("""
            **Mengapa Memilih Random Forest untuk Tahap Deployment?**
            
            Berdasarkan komparasi performa dan stabilitas, **Random Forest** direkomendasikan sebagai arsitektur utama aplikasi cerdas ini. 
            
            Sebagai metode *Ensemble*, algoritma ini bekerja dengan membangun sekumpulan pohon keputusan (100 *decision trees*) lalu mengambil keputusan berbasis mayoritas suara (*majority voting*). Pendekatan ini membuatnya jauh lebih kuat terhadap *noise* data log Microsoft Teams dan memiliki resiko *overfitting* yang jauh lebih kecil dibanding pohon keputusan tunggal.
            """)
            
        st.markdown("---")
        
        # Feature Importance
        st.markdown("### 🔑 Nilai Kepentingan Fitur (Feature Importance) - Model Random Forest")
        importances = eval_dict['Objects'][1].feature_importances_
        df_imp = pd.DataFrame({'Fitur': fitur_names, 'Bobot Kepentingan': importances}).sort_values(by='Bobot Kepentingan', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Bobot Kepentingan', y='Fitur', orientation='h',
                         title="Atribut yang Paling Berpengaruh dalam Keputusan AI",
                         color_discrete_sequence=['#E91E63'])
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # PENJELASAN VISUALISASI
        st.markdown("""
        **📌 Penjelasan Visualisasi Bobot Fitur:** Grafik ini membongkar aspek internal cara berpikir kecerdasan buatan dalam melakukan pengelompokan. Atribut dengan nilai tertinggi (berada di posisi paling atas) menunjukkan indikator utama yang paling sensitif dalam menggeser status pengguna dari tingkat Rendah menuju Sedang ataupun Tinggi.
        """)

    # ----------------------------------------
    # TAB 4: SIMULASI PREDIKSI (DEPLOYMENT)
    # ----------------------------------------
    with tab4:
        # HEADER BANNER
        st.markdown("""
        <div style="background-color:#D81B60;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🚀 SIMULASI PREDIKSI OTOMATIS (DEPLOYMENT INTERAKTIF)</h2>
            <p style="color:#FCE4EC;margin:5px 0 0 0">Implementasi Model Machine Learning Terpilih untuk Mengklasifikasikan Data Pengguna Baru</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PANDUAN INFORMASI
        st.info("""
        ℹ️ **Panduan Simulasi Prediksi:** 1. Masukkan estimasi akumulasi durasi penggunaan Microsoft Teams milik seorang Dosen atau Tendik dalam satuan **Menit** pada kolom input di bawah ini.
        2. Klik tombol **"Mulai Analisis Kecerdasan Buatan"** untuk memerintahkan model *Random Forest* melakukan komputasi prediksi.
        3. Sistem akan secara otomatis menyelaraskan skala data (*Z-score scaling*) sesuai dengan pola distribusi data latih kampus sebelum mengeluarkan hasil akhir.
        """)
        
        with st.form("form_prediksi"):
            st.markdown("#### 📥 Form Input Data Aktivitas Pengguna (Dalam Satuan Menit)")
            col_in1, col_in2, col_in3 = st.columns(3)
            
            with col_in1:
                audio_in = st.number_input("🎙️ Durasi Berbicara/Audio (Menit)", min_value=0, value=830, step=10, help="Total waktu pengguna berinteraksi menggunakan suara/microphone.")
            with col_in2:
                video_in = st.number_input("📹 Durasi Menyalakan Kamera/Video (Menit)", min_value=0, value=750, step=10, help="Total waktu pengguna mengaktifkan webcam/kamera visual.")
            with col_in3:
                screen_in = st.number_input("💻 Durasi Berbagi Layar/Screen Share (Menit)", min_value=0, value=500, step=10, help="Total waktu pengguna membagikan layar presentasi/dokumen.")
                
            submit_btn = st.form_submit_button("Mulai Analisis Kecerdasan Buatan", type="primary")
            
        if submit_btn:
            total_in = audio_in + video_in + screen_in
            data_baru = pd.DataFrame({
                'Audio Duration (Menit)': [audio_in],
                'Video Duration (Menit)': [video_in],
                'Screen Share (Menit)': [screen_in],
                'Total_Duration (Menit)': [total_in]
            })
            
            # Melakukan transformasi penskalaan fitur
            data_baru_scaled = scaler.transform(data_baru)
            
            # Prediksi kelas dan probabilitas
            pred_kode = model_terpilih.predict(data_baru_scaled)[0]
            hasil_prediksi = inv_map[pred_kode]
            proba = model_terpilih.predict_proba(data_baru_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🔔 Hasil Analisis Sistem AI")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Total Durasi Terkalkulasi", f"{total_in} Menit")
                # KETERANGAN PREDIKSI YANG JELAS (KATEGORI HASIL)
                if hasil_prediksi == "Rendah":
                    st.markdown("Tingkat Klasifikasi: <span style='color:#EF5350;font-weight:bold;font-size:24px'>RENDAH</span>", unsafe_allow_html=True)
                elif hasil_prediksi == "Sedang":
                    st.markdown("Tingkat Klasifikasi: <span style='color:#FFCA28;font-weight:bold;font-size:24px'>SEDANG</span>", unsafe_allow_html=True)
                else:
                    st.markdown("Tingkat Klasifikasi: <span style='color:#66BB6A;font-weight:bold;font-size:24px'>TINGGI</span>", unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("**📊 Tingkat Keyakinan Keputusan Model (*Class Probabilities*):**")
                st.progress(float(proba[0]), text=f"Kemungkinan Kelas Rendah: {proba[0]:.1%}")
                st.progress(float(proba[1]), text=f"Kemungkinan Kelas Sedang: {proba[1]:.1%}")
                st.progress(float(proba[2]), text=f"Kemungkinan Kelas Tinggi: {proba[2]:.1%}")
            
            # KETERANGAN PREDIKSI YANG JELAS (INTERPRETASI & REKOMENDASI OPERASIONAL)
            st.markdown("### 📌 Arti Hasil Klasifikasi & Rekomendasi Kebijakan Kampus:")
            if hasil_prediksi == "Rendah":
                st.error("""
                **Interpretasi Kelas - RENDAH:** Pengguna yang masuk dalam ketegori ini tercatat memiliki interaksi yang sangat minim pada platform Microsoft Teams selama periode observasi. 
                
                **Rekomendasi Tindakan Administrasi Kampus UIN Jakarta:** * Perlu adanya sosialisasi atau pelatihan penyegaran (*refreshment training*) terkait pemanfaatan fitur e-learning dan kerja kolaboratif digital dari Pusat Teknologi Informasi dan Pangkalan Data (PUSTIPANDA).
                * Fakultas direkomendasikan untuk melakukan pengecekan apakah pengguna mengalami kendala teknis (seperti masalah akun atau keterbatasan perangkat).
                """)
            elif hasil_prediksi == "Sedang":
                st.warning("""
                **Interpretasi Kelas - SEDANG:** Pengguna dalam kategori ini dinilai sudah beradaptasi dengan baik dalam ekosistem digital kampus. Aktivitas komunikasi suara berjalan rutin, namun pemanfaatan fitur interaksi tingkat lanjut seperti berbagi layar (*screen share*) atau visualisasi kamera (*video*) masih dapat dioptimalkan.
                
                **Rekomendasi Tindakan Administrasi Kampus UIN Jakarta:** * Dorong pengguna untuk mengaktifkan kamera pada sesi rapat krusial guna meningkatkan interaksi yang lebih intim dan profesional.
                * Berikan apresiasi berkala agar pengguna mempertahankan serta meningkatkan ritme kolaborasinya.
                """)
            else:
                st.success("""
                **Interpretasi Kelas - TINGGI:** Pengguna diklasifikasikan sebagai *Power User* atau pengguna yang sangat aktif. Durasi audio, video, dan intensitas presentasi pembagian layar yang tercatat berada di atas rata-rata civitas akademika lainnya, mencerminkan komitmen tinggi terhadap pelaksanaan kegiatan jarak jauh maupun administrasi berbasis digital.
                
                **Rekomendasi Tindakan Administrasi Kampus UIN Jakarta:** * Sangat direkomendasikan untuk dijadikan *Role Model* atau mentor sejawat (*peer mentor*) di lingkungan unit kerja atau fakultas masing-masing guna menularkan budaya kerja digital yang efisien.
                * Profil pengguna ini dapat dilibatkan dalam kelompok penguji (*beta tester*) jika kampus berencana menerapkan modul atau sistem aplikasi baru.
                """)
