import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    mhs_file = 'mhs juni.csv'
    staff_file = 'staff juni.csv'
    
    if not os.path.exists(mhs_file) or not os.path.exists(staff_file):
        return None, None
        
    df_mhs = pd.read_csv(mhs_file)
    df_staff = pd.read_csv(staff_file)
    
    def proses_durasi(df):
        # Konversi Detik ke Menit
        df['Audio Duration (Menit)'] = df['Audio Duration In Seconds'] / 60
        df['Video Duration (Menit)'] = df['Video Duration In Seconds'] / 60
        df['Screen Share (Menit)'] = df['Screen Share Duration In Seconds'] / 60
        df['Total_Duration (Menit)'] = (
            df['Audio Duration (Menit)'] + 
            df['Video Duration (Menit)'] + 
            df['Screen Share (Menit)']
        )
        
        # Membuat Label Target berdasarkan Meeting Count
        def kategori_aktivitas(x):
            if x <= 5: return "Rendah"
            elif x <= 15: return "Sedang"
            else: return "Tinggi"
            
        df['Activity_Level'] = df['Meeting Count'].apply(kategori_aktivitas)
        
        # Handling missing names untuk leaderboard
        if 'Nama' in df.columns:
            df['Nama_Tampil'] = df['Nama'].fillna(df['Username'])
        else:
            df['Nama_Tampil'] = df['Username']
            
        return df

    # 1. Menangani Data Mahasiswa
    df_mhs = df_mhs.dropna(subset=['Last Activity Date'])
    df_mhs = proses_durasi(df_mhs)
    # Membuat ID Pengguna Anonim untuk preview tabel
    df_mhs.insert(0, 'User ID', ['MHS_' + str(i).zfill(5) for i in range(1, len(df_mhs) + 1)])
    
    # 2. Menangani Data Staff (Dosen & Tendik)
    df_staff = df_staff.dropna(subset=['Last Activity Date'])
    df_staff['Role'] = df_staff['Role'].fillna('Tendik')  # Asumsi default jika role kosong
    df_staff['Role'] = df_staff['Role'].replace({'tendik': 'Tendik', 'dosen': 'Dosen'})
    df_staff = proses_durasi(df_staff)
    # Membuat ID Pengguna Anonim untuk preview tabel
    df_staff.insert(0, 'User ID', ['STAFF_' + str(i).zfill(4) for i in range(1, len(df_staff) + 1)])
    
    return df_mhs, df_staff

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Microsoft_Teams_14_logo.svg/512px-Microsoft_Teams_14_logo.svg.png", width=80)
st.sidebar.title("MS Teams Analytics")
st.sidebar.markdown("Dashboard Aktivitas MS Teams UIN Jakarta (Periode Juni)")
st.sidebar.markdown("---")
st.sidebar.success("🔒 **Data Privacy Hybrid**\nIdentitas disamarkan (ID Anonim) pada ringkasan umum, namun Papan Peringkat (Leaderboard) menampilkan otoritas nama asli untuk apresiasi/evaluasi institusi.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Instansi:** UIN Syarif Hidayatullah Jakarta")

# ==========================================
# PEMBACAAN DATA UTAMA & UI
# ==========================================
df_mhs, df_staff = load_and_preprocess_data()

if df_mhs is None or df_staff is None:
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.error("⚠️ File `mhs juni.csv` dan/atau `staff juni.csv` tidak ditemukan. Pastikan kedua file tersebut berada di folder yang sama dengan file aplikasi ini.")
else:
    # JUDUL UTAMA
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.markdown("### Laporan Periode Bulan Juni | UIN Syarif Hidayatullah Jakarta")
    st.markdown("---")
    
    # MEMBUAT 3 TAB UTAMA (SESUAI PERMINTAAN)
    tab1, tab2, tab3 = st.tabs([
        "📝 Ringkasan Informasi", 
        "🎓 EDA Mahasiswa", 
        "💼 EDA Dosen dan Tendik"
    ])
    
    # ----------------------------------------
    # TAB 1: RINGKASAN INFORMASI
    # ----------------------------------------
    with tab1:
        st.markdown("""
        <div style="background-color:#004D40;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">📝 RINGKASAN EKSEKUTIF (BULAN JUNI)</h2>
            <p style="color:#E0F2F1;margin:5px 0 0 0">Pusat Informasi Utama dan Statistik Dasar Penggunaan MS Teams Keseluruhan</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ringkasan KPI
        st.markdown("### 📊 Indikator Kunci (Pengguna Aktif Bulan Juni)")
        col1, col2, col3 = st.columns(3)
        total_mhs = len(df_mhs)
        total_dosen = len(df_staff[df_staff['Role'].str.contains('Dosen', case=False, na=False)])
        total_tendik = len(df_staff) - total_dosen
        
        col1.metric("Mahasiswa Aktif", f"{total_mhs:,}".replace(',', '.'))
        col2.metric("Dosen Aktif", f"{total_dosen:,}".replace(',', '.'))
        col3.metric("Tendik/Lainnya Aktif", f"{total_tendik:,}".replace(',', '.'))
        
        st.markdown("---")
        
        # Rata-rata Durasi Komparasi
        col_avg1, col_avg2 = st.columns(2)
        with col_avg1:
            st.markdown("#### 🎓 Rata-rata Durasi (Mahasiswa)")
            st.info(f"🎙️ Audio: **{df_mhs['Audio Duration (Menit)'].mean():.1f}** Menit\n\n📹 Video: **{df_mhs['Video Duration (Menit)'].mean():.1f}** Menit\n\n💻 Screen Share: **{df_mhs['Screen Share (Menit)'].mean():.1f}** Menit")
            
        with col_avg2:
            st.markdown("#### 💼 Rata-rata Durasi (Staff / Dosen)")
            st.success(f"🎙️ Audio: **{df_staff['Audio Duration (Menit)'].mean():.1f}** Menit\n\n📹 Video: **{df_staff['Video Duration (Menit)'].mean():.1f}** Menit\n\n💻 Screen Share: **{df_staff['Screen Share (Menit)'].mean():.1f}** Menit")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Preview Data (Anonim)
        st.markdown("### 📋 Preview Dataset Mahasiswa (Telah Dianonimkan)")
        kolom_tampil_mhs = ['User ID', 'Fakultas', 'Prodi', 'Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)', 'Activity_Level']
        st.dataframe(df_mhs[kolom_tampil_mhs].head(5), use_container_width=True, hide_index=True)
        
        st.markdown("### 📋 Preview Dataset Staff (Telah Dianonimkan)")
        kolom_tampil_staff = ['User ID', 'Role', 'Unit Kerja', 'Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)', 'Activity_Level']
        st.dataframe(df_staff[kolom_tampil_staff].head(5), use_container_width=True, hide_index=True)

    # ----------------------------------------
    # TAB 2: EDA MAHASISWA
    # ----------------------------------------
    with tab2:
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🎓 EXPLORATORY DATA ANALYSIS (MAHASISWA)</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Eksplorasi Karakteristik dan Aktivitas Mahasiswa Berdasarkan Fakultas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter MHS
        st.markdown("### 🔍 Filter Data Mahasiswa")
        list_fakultas = ["Semua Fakultas"] + sorted(df_mhs['Fakultas'].dropna().unique().tolist())
        pilihan_fakultas = st.selectbox("Pilih Fakultas untuk Dianalisis:", list_fakultas, key="fakultas_mhs")
        
        if pilihan_fakultas == "Semua Fakultas":
            df_mhs_eda = df_mhs.copy()
        else:
            df_mhs_eda = df_mhs[df_mhs['Fakultas'] == pilihan_fakultas].copy()
            
        st.markdown("---")
        
        # Visualisasi MHS
        col_mhs1, col_mhs2 = st.columns(2)
        
        with col_mhs1:
            act_counts_mhs = df_mhs_eda['Activity_Level'].value_counts().reset_index()
            act_counts_mhs.columns = ['Activity_Level', 'Jumlah']
            act_counts_mhs['Activity_Level'] = pd.Categorical(act_counts_mhs['Activity_Level'], categories=["Rendah", "Sedang", "Tinggi"], ordered=True)
            act_counts_mhs = act_counts_mhs.sort_values('Activity_Level')
            
            fig_bar_mhs = px.bar(act_counts_mhs, x='Activity_Level', y='Jumlah', text='Jumlah', 
                             color='Activity_Level', 
                             color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
                             title=f"Distribusi Tingkat Aktivitas (Mahasiswa)")
            fig_bar_mhs.update_layout(showlegend=False)
            st.plotly_chart(fig_bar_mhs, use_container_width=True)
            
        with col_mhs2:
            avg_mhs_data = {
                'Jenis Fitur': ['Audio (Suara)', 'Video (Kamera)', 'Screen Share'],
                'Rata-rata (Menit)': [
                    df_mhs_eda['Audio Duration (Menit)'].mean(),
                    df_mhs_eda['Video Duration (Menit)'].mean(),
                    df_mhs_eda['Screen Share (Menit)'].mean()
                ]
            }
            # Menghindari error pembagian jika filter kosong
            max_avg_mhs = max(avg_mhs_data['Rata-rata (Menit)']) if pd.notna(max(avg_mhs_data['Rata-rata (Menit)'])) else 1
            
            fig_avg_mhs = px.bar(
                pd.DataFrame(avg_mhs_data), x='Jenis Fitur', y='Rata-rata (Menit)', text='Rata-rata (Menit)',
                color='Jenis Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'],
                title=f"Rata-rata Durasi Fitur (Mahasiswa)"
            )
            fig_avg_mhs.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_avg_mhs.update_layout(showlegend=False, yaxis_range=[0, max_avg_mhs * 1.3])
            st.plotly_chart(fig_avg_mhs, use_container_width=True)
            
        st.markdown("---")
        
        # Papan Peringkat MHS
        st.markdown("### 🏆 Papan Peringkat Mahasiswa Teraktif (Top 10)")
        
        tab_mhs_meet, tab_mhs_aud, tab_mhs_vid = st.tabs(["📊 Frekuensi Rapat Terbanyak", "🎙️ Durasi Audio Terlama", "📹 Durasi Video Terlama"])
        
        with tab_mhs_meet:
            top_mhs_meet = df_mhs_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count', ascending=True)
            fig_mhs_meet = px.bar(top_mhs_meet, x='Meeting Count', y='Nama_Tampil', orientation='h', 
                              text='Meeting Count', title="Top 10: Frekuensi Pertemuan/Kelas",
                              color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig_mhs_meet, use_container_width=True)

        with tab_mhs_aud:
            top_mhs_aud = df_mhs_eda.nlargest(10, 'Audio Duration (Menit)').sort_values('Audio Duration (Menit)', ascending=True)
            fig_mhs_aud = px.bar(top_mhs_aud, x='Audio Duration (Menit)', y='Nama_Tampil', orientation='h', 
                             text='Audio Duration (Menit)', title="Top 10: Partisipasi Suara Terlama",
                             color_discrete_sequence=['#2196F3'])
            fig_mhs_aud.update_traces(texttemplate='%{text:.0f}')
            st.plotly_chart(fig_mhs_aud, use_container_width=True)

        with tab_mhs_vid:
            top_mhs_vid = df_mhs_eda.nlargest(10, 'Video Duration (Menit)').sort_values('Video Duration (Menit)', ascending=True)
            fig_mhs_vid = px.bar(top_mhs_vid, x='Video Duration (Menit)', y='Nama_Tampil', orientation='h', 
                             text='Video Duration (Menit)', title="Top 10: Kamera Menyala Terlama",
                             color_discrete_sequence=['#E91E63'])
            fig_mhs_vid.update_traces(texttemplate='%{text:.0f}')
            st.plotly_chart(fig_mhs_vid, use_container_width=True)
            
    # ----------------------------------------
    # TAB 3: EDA DOSEN DAN TENDIK
    # ----------------------------------------
    with tab3:
        st.markdown("""
        <div style="background-color:#FF8F00;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">💼 EXPLORATORY DATA ANALYSIS (DOSEN & TENDIK)</h2>
            <p style="color:#FFF8E1;margin:5px 0 0 0">Eksplorasi Karakteristik dan Aktivitas Organisasional Staff (Berdasarkan Unit & Peran)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter Staff
        st.markdown("### 🔍 Filter Data Staff")
        col_stf1, col_stf2 = st.columns(2)
        with col_stf1:
            list_roles = ["Semua Peran"] + sorted(df_staff['Role'].dropna().unique().tolist())
            pilihan_role = st.selectbox("Pilih Klasifikasi Peran:", list_roles, key="role_staff")
        with col_stf2:
            list_unit = ["Semua Unit Kerja"] + sorted(df_staff['Unit Kerja'].dropna().unique().tolist())
            pilihan_unit = st.selectbox("Pilih Unit Kerja (Fakultas/Lembaga):", list_unit, key="unit_staff")
            
        # Mengaplikasikan filter
        df_staff_eda = df_staff.copy()
        if pilihan_role != "Semua Peran":
            df_staff_eda = df_staff_eda[df_staff_eda['Role'] == pilihan_role]
        if pilihan_unit != "Semua Unit Kerja":
            df_staff_eda = df_staff_eda[df_staff_eda['Unit Kerja'] == pilihan_unit]
            
        st.markdown("---")
        
        # Visualisasi Staff
        col_stf_v1, col_stf_v2 = st.columns(2)
        
        with col_stf_v1:
            # Jika user memilih semua peran, tampilkan chart proporsi peran. Jika tidak, tampilkan Activity Level.
            if pilihan_role == "Semua Peran":
                role_counts = df_staff_eda['Role'].value_counts().reset_index()
                role_counts.columns = ['Role', 'Jumlah']
                fig_pie_staff = px.pie(role_counts, values='Jumlah', names='Role', hole=0.5, 
                                 color_discrete_sequence=CUSTOM_COLORS, title="Komposisi Staff Aktif (Dosen vs Tendik)")
                fig_pie_staff.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie_staff, use_container_width=True)
            else:
                act_counts_stf = df_staff_eda['Activity_Level'].value_counts().reset_index()
                act_counts_stf.columns = ['Activity_Level', 'Jumlah']
                act_counts_stf['Activity_Level'] = pd.Categorical(act_counts_stf['Activity_Level'], categories=["Rendah", "Sedang", "Tinggi"], ordered=True)
                act_counts_stf = act_counts_stf.sort_values('Activity_Level')
                
                fig_bar_stf = px.bar(act_counts_stf, x='Activity_Level', y='Jumlah', text='Jumlah', 
                                 color='Activity_Level', 
                                 color_discrete_map={"Rendah": "#EF5350", "Sedang": "#FFCA28", "Tinggi": "#66BB6A"},
                                 title=f"Distribusi Tingkat Aktivitas Staff")
                fig_bar_stf.update_layout(showlegend=False)
                st.plotly_chart(fig_bar_stf, use_container_width=True)
                
        with col_stf_v2:
            avg_stf_data = {
                'Jenis Fitur': ['Audio (Suara)', 'Video (Kamera)', 'Screen Share'],
                'Rata-rata (Menit)': [
                    df_staff_eda['Audio Duration (Menit)'].mean(),
                    df_staff_eda['Video Duration (Menit)'].mean(),
                    df_staff_eda['Screen Share (Menit)'].mean()
                ]
            }
            max_avg_stf = max(avg_stf_data['Rata-rata (Menit)']) if pd.notna(max(avg_stf_data['Rata-rata (Menit)'])) else 1
            
            fig_avg_stf = px.bar(
                pd.DataFrame(avg_stf_data), x='Jenis Fitur', y='Rata-rata (Menit)', text='Rata-rata (Menit)',
                color='Jenis Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'],
                title="Rata-rata Durasi Fitur (Staff)"
            )
            fig_avg_stf.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_avg_stf.update_layout(showlegend=False, yaxis_range=[0, max_avg_stf * 1.3])
            st.plotly_chart(fig_avg_stf, use_container_width=True)

        st.markdown("---")
        
        # Papan Peringkat Staff
        st.markdown("### 🏆 Papan Peringkat Dosen & Tendik Teraktif (Top 10)")
        
        tab_stf_meet, tab_stf_aud, tab_stf_vid = st.tabs(["📊 Frekuensi Rapat Terbanyak", "🎙️ Durasi Audio Terlama", "💻 Durasi Screen Share (Presentasi)"])
        
        if len(df_staff_eda) > 0:
            with tab_stf_meet:
                top_stf_meet = df_staff_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count', ascending=True)
                fig_stf_meet = px.bar(top_stf_meet, x='Meeting Count', y='Nama_Tampil', color='Role', orientation='h', 
                                  text='Meeting Count', title="Top 10: Penyelenggara/Peserta Rapat Terbanyak")
                st.plotly_chart(fig_stf_meet, use_container_width=True)

            with tab_stf_aud:
                top_stf_aud = df_staff_eda.nlargest(10, 'Audio Duration (Menit)').sort_values('Audio Duration (Menit)', ascending=True)
                fig_stf_aud = px.bar(top_stf_aud, x='Audio Duration (Menit)', y='Nama_Tampil', color='Role', orientation='h', 
                                 text='Audio Duration (Menit)', title="Top 10: Komunikasi Suara Terlama")
                fig_stf_aud.update_traces(texttemplate='%{text:.0f}')
                st.plotly_chart(fig_stf_aud, use_container_width=True)

            with tab_stf_vid:
                top_stf_scr = df_staff_eda.nlargest(10, 'Screen Share (Menit)').sort_values('Screen Share (Menit)', ascending=True)
                fig_stf_scr = px.bar(top_stf_scr, x='Screen Share (Menit)', y='Nama_Tampil', color='Role', orientation='h', 
                                 text='Screen Share (Menit)', title="Top 10: Pemateri/Presentasi Layar Terlama")
                fig_stf_scr.update_traces(texttemplate='%{text:.0f}')
                st.plotly_chart(fig_stf_scr, use_container_width=True)
        else:
            st.warning("Tidak ada data pengguna yang memenuhi kriteria filter yang Anda pilih.")