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
        return None, None, None
        
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
        def golongkan_aktivitas(row):
            if pd.isna(row['Last Activity Date DT']):
                return "Tidak Aktif (Dalam 180 Hari)"
            hari = row['Hari Sejak Akses Terakhir']
            if hari <= 7:
                return "Sangat Aktif (Akses 0-7 Hari Lalu)"
            elif hari <= 30:
                return "Aktif (Akses 8-30 Hari Lalu)"
            elif hari <= 90:
                return "Cukup Aktif (Akses 31-90 Hari Lalu)"
            else:
                return "Pasif (Akses >90 Hari Lalu)"
                
        df['Tingkat_Aktivitas_Recency'] = df.apply(golongkan_aktivitas, axis=1)
        
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
    
    return df_mhs, df_dosen, df_tendik

# ==========================================
# PEMBACAAN DATA UTAMA & UI
# ==========================================
df_mhs, df_dosen, df_tendik = load_and_preprocess_data()

if df_mhs is None or df_dosen is None or df_tendik is None:
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.error("⚠️ File `mhs juni.csv` dan/atau `staff juni.csv` tidak ditemukan di direktori aplikasi.")
else:
    # Header Aplikasi Utama
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.markdown("### Laporan Periode Akumulasi 180 Hari (6 Bulan) | UIN Syarif Hidayatullah Jakarta")
    st.markdown("---")
    
    # Navigasi Empat Tab Utama
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Ringkasan", 
        "🎓 EDA Mahasiswa", 
        "👨‍🏫 EDA Dosen",
        "💼 EDA Tendik"
    ])
    
    status_order = ["Sangat Aktif (Akses 0-7 Hari Lalu)", "Aktif (Akses 8-30 Hari Lalu)", "Cukup Aktif (Akses 31-90 Hari Lalu)", "Pasif (Akses >90 Hari Lalu)", "Tidak Aktif (Dalam 180 Hari)"]
    
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
        col1.metric("Total Lisensi Mahasiswa Terdata", f"{len(df_mhs):,}".replace(',', '.'))
        col2.metric("Total Lisensi Dosen Terdata", f"{len(df_dosen):,}".replace(',', '.'))
        col3.metric("Total Lisensi Tendik Terdata", f"{len(df_tendik):,}".replace(',', '.'))
        
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

    # ----------------------------------------
    # TAB 2: EDA MAHASISWA
    # ----------------------------------------
    with tab2:
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🎓 EXPLORATORY DATA ANALYSIS: MAHASISWA</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Analisis Perilaku Digital, Retensi Akses, dan Kinerja Berdasarkan Fakultas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter MHS
        list_fakultas = ["Semua Fakultas"] + sorted(df_mhs['Fakultas'].dropna().unique().tolist())
        pilihan_fakultas = st.selectbox("🔍 Filter Fakultas (Mahasiswa):", list_fakultas, key="filter_mhs")
        df_mhs_eda = df_mhs.copy() if pilihan_fakultas == "Semua Fakultas" else df_mhs[df_mhs['Fakultas'] == pilihan_fakultas].copy()
        
        st.markdown("---")
        
        # Visualisasi MHS
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mhs_status = df_mhs_eda['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
            mhs_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah']
            fig_mhs_rec = px.bar(mhs_status, x='Tingkat Aktivitas Resensi', y='Jumlah', text='Jumlah', title="Distribusi Status Akses (Resensi)", color='Tingkat Aktivitas Resensi', color_discrete_sequence=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'])
            fig_mhs_rec.update_layout(showlegend=False)
            st.plotly_chart(fig_mhs_rec, use_container_width=True)
            st.caption("**Interpretasi:** Melihat seberapa banyak mahasiswa yang benar-benar menggunakan Teams pada kurun waktu terdekat untuk evaluasi lisensi IT.")
            
        with col_m2:
            avg_mhs = pd.DataFrame({
                'Fitur': ['Audio', 'Video', 'Screen Share'],
                'Rata-rata (Jam)': [df_mhs_eda['Audio Duration (Jam)'].mean(), df_mhs_eda['Video Duration (Jam)'].mean(), df_mhs_eda['Screen Share (Jam)'].mean()]
            })
            fig_avg_mhs = px.bar(avg_mhs, x='Fitur', y='Rata-rata (Jam)', text='Rata-rata (Jam)', title="Rata-rata Durasi Fitur per Mahasiswa", color='Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'])
            fig_avg_mhs.update_traces(texttemplate='%{text:.2f} Jam')
            fig_avg_mhs.update_layout(showlegend=False)
            st.plotly_chart(fig_avg_mhs, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🏛️ Peringkat Fakultas (Agregat Total)")
        df_fakultas_grp = df_mhs_eda.groupby('Fakultas')[['Meeting Count', 'Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)']].sum().reset_index()
        
        t1, t2, t3, t4 = st.tabs(["📊 Pertemuan (Rapat)", "🎙️ Durasi Audio", "📹 Durasi Video", "💻 Durasi Screen Share"])
        with t1:
            st.plotly_chart(px.bar(df_fakultas_grp.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Fakultas', orientation='h', text_auto='.0f', title='Top 10 Fakultas (Akumulasi Meeting)'), use_container_width=True)
        with t2:
            st.plotly_chart(px.bar(df_fakultas_grp.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Fakultas', orientation='h', text_auto='.1f', title='Top 10 Fakultas (Total Jam Audio)'), use_container_width=True)
        with t3:
            st.plotly_chart(px.bar(df_fakultas_grp.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Fakultas', orientation='h', text_auto='.1f', title='Top 10 Fakultas (Total Jam Video)'), use_container_width=True)
        with t4:
            st.plotly_chart(px.bar(df_fakultas_grp.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Fakultas', orientation='h', text_auto='.1f', title='Top 10 Fakultas (Total Jam Screen Share)'), use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🏆 Top 10 Individu Mahasiswa Teraktif")
        t_i1, t_i2, t_i3, t_i4 = st.tabs(["📊 Frekuensi Rapat", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with t_i1:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Mahasiswa: Frekuensi Kelas Virtual Terbanyak'), use_container_width=True)
        with t_i2:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Durasi Audio (Jam) Terlama'), use_container_width=True)
        with t_i3:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Durasi Video (Jam) Terlama'), use_container_width=True)
        with t_i4:
            st.plotly_chart(px.bar(df_mhs_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Mahasiswa: Durasi Screen Share (Jam) Terlama'), use_container_width=True)

    # ----------------------------------------
    # TAB 3: EDA DOSEN
    # ----------------------------------------
    with tab3:
        st.markdown("""
        <div style="background-color:#FF8F00;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">👨‍🏫 EXPLORATORY DATA ANALYSIS: DOSEN</h2>
            <p style="color:#FFF8E1;margin:5px 0 0 0">Analisis Kinerja Pengajaran, Konsistensi Akses Dosen, dan Distribusi Beban Unit Kerja</p>
        </div>
        """, unsafe_allow_html=True)
        
        list_unit_dosen = ["Semua Unit Kerja"] + sorted(df_dosen['Unit Kerja'].dropna().unique().tolist())
        pilihan_unit_dosen = st.selectbox("🔍 Filter Unit Kerja (Dosen):", list_unit_dosen, key="filter_dosen")
        df_dosen_eda = df_dosen.copy() if pilihan_unit_dosen == "Semua Unit Kerja" else df_dosen[df_dosen['Unit Kerja'] == pilihan_unit_dosen].copy()
        
        st.markdown("---")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dsn_status = df_dosen_eda['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
            dsn_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah']
            fig_dsn_rec = px.bar(dsn_status, x='Tingkat Aktivitas Resensi', y='Jumlah', text='Jumlah', title="Distribusi Status Akses Dosen (Resensi)", color='Tingkat Aktivitas Resensi', color_discrete_sequence=['#004D40', '#00796B', '#80CBC4', '#FFB74D', '#E53935'])
            fig_dsn_rec.update_layout(showlegend=False)
            st.plotly_chart(fig_dsn_rec, use_container_width=True)
            st.caption("**Interpretasi:** Menunjukkan konsistensi dosen dalam memanfaatkan Teams untuk pembelajaran. Angka 'Tidak Aktif' yang tinggi dapat menjadi indikasi perlunya pelatihan platform terpadu.")
            
        with col_d2:
            avg_dsn = pd.DataFrame({
                'Fitur': ['Audio', 'Video', 'Screen Share'],
                'Rata-rata (Jam)': [df_dosen_eda['Audio Duration (Jam)'].mean(), df_dosen_eda['Video Duration (Jam)'].mean(), df_dosen_eda['Screen Share (Jam)'].mean()]
            })
            fig_avg_dsn = px.bar(avg_dsn, x='Fitur', y='Rata-rata (Jam)', text='Rata-rata (Jam)', title="Rata-rata Durasi Fitur per Dosen", color='Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'])
            fig_avg_dsn.update_traces(texttemplate='%{text:.2f} Jam')
            fig_avg_dsn.update_layout(showlegend=False)
            st.plotly_chart(fig_avg_dsn, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🏛️ Peringkat Unit Kerja Dosen (Agregat Total)")
        df_dsn_grp = df_dosen_eda.groupby('Unit Kerja')[['Meeting Count', 'Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)']].sum().reset_index()
        
        t_d1, t_d2, t_d3, t_d4 = st.tabs(["📊 Pertemuan (Rapat)", "🎙️ Durasi Audio", "📹 Durasi Video", "💻 Durasi Screen Share"])
        with t_d1:
            st.plotly_chart(px.bar(df_dsn_grp.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit (Akumulasi Meeting Dosen)'), use_container_width=True)
        with t_d2:
            st.plotly_chart(px.bar(df_dsn_grp.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Audio Dosen)'), use_container_width=True)
        with t_d3:
            st.plotly_chart(px.bar(df_dsn_grp.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Video Dosen)'), use_container_width=True)
        with t_d4:
            st.plotly_chart(px.bar(df_dsn_grp.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Screen Share Dosen)'), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏆 Top 10 Individu Dosen Teraktif")
        td_i1, td_i2, td_i3, td_i4 = st.tabs(["📊 Frekuensi Mengajar/Rapat", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with td_i1:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Dosen: Frekuensi Kelas/Rapat Terbanyak'), use_container_width=True)
        with td_i2:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Durasi Audio (Jam) Terlama'), use_container_width=True)
        with td_i3:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Durasi Video (Jam) Terlama'), use_container_width=True)
        with td_i4:
            st.plotly_chart(px.bar(df_dosen_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Dosen: Durasi Screen Share (Jam) Terlama'), use_container_width=True)

    # ----------------------------------------
    # TAB 4: EDA TENDIK
    # ----------------------------------------
    with tab4:
        st.markdown("""
        <div style="background-color:#6A1B9A;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">💼 EXPLORATORY DATA ANALYSIS: TENDIK</h2>
            <p style="color:#F3E5F5;margin:5px 0 0 0">Analisis Beban Koordinasi Administratif dan Resensi Tenaga Kependidikan</p>
        </div>
        """, unsafe_allow_html=True)
        
        list_unit_tendik = ["Semua Unit Kerja"] + sorted(df_tendik['Unit Kerja'].dropna().unique().tolist())
        pilihan_unit_tendik = st.selectbox("🔍 Filter Unit Kerja (Tendik):", list_unit_tendik, key="filter_tendik")
        df_tendik_eda = df_tendik.copy() if pilihan_unit_tendik == "Semua Unit Kerja" else df_tendik[df_tendik['Unit Kerja'] == pilihan_unit_tendik].copy()
        
        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tdk_status = df_tendik_eda['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
            tdk_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah']
            fig_tdk_rec = px.bar(tdk_status, x='Tingkat Aktivitas Resensi', y='Jumlah', text='Jumlah', title="Distribusi Status Akses Tendik (Resensi)", color='Tingkat Aktivitas Resensi', color_discrete_sequence=['#4A148C', '#7B1FA2', '#BA68C8', '#FFB74D', '#E53935'])
            fig_tdk_rec.update_layout(showlegend=False)
            st.plotly_chart(fig_tdk_rec, use_container_width=True)
            st.caption("**Interpretasi:** Mendeteksi unit administrasi/biro yang sangat proaktif berkoordinasi secara virtual. Tendik idealnya menempati zona 'Aktif' untuk menjamin kelancaran layanan institusi.")
            
        with col_t2:
            avg_tdk = pd.DataFrame({
                'Fitur': ['Audio', 'Video', 'Screen Share'],
                'Rata-rata (Jam)': [df_tendik_eda['Audio Duration (Jam)'].mean(), df_tendik_eda['Video Duration (Jam)'].mean(), df_tendik_eda['Screen Share (Jam)'].mean()]
            })
            fig_avg_tdk = px.bar(avg_tdk, x='Fitur', y='Rata-rata (Jam)', text='Rata-rata (Jam)', title="Rata-rata Durasi Fitur per Tendik", color='Fitur', color_discrete_sequence=['#1E88E5', '#D81B60', '#FFC107'])
            fig_avg_tdk.update_traces(texttemplate='%{text:.2f} Jam')
            fig_avg_tdk.update_layout(showlegend=False)
            st.plotly_chart(fig_avg_tdk, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🏛️ Peringkat Unit Kerja Tendik (Agregat Total)")
        df_tdk_grp = df_tendik_eda.groupby('Unit Kerja')[['Meeting Count', 'Audio Duration (Jam)', 'Video Duration (Jam)', 'Screen Share (Jam)']].sum().reset_index()
        
        tt_d1, tt_d2, tt_d3, tt_d4 = st.tabs(["📊 Pertemuan (Rapat)", "🎙️ Durasi Audio", "📹 Durasi Video", "💻 Durasi Screen Share"])
        with tt_d1:
            st.plotly_chart(px.bar(df_tdk_grp.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit (Akumulasi Meeting Tendik)'), use_container_width=True)
        with tt_d2:
            st.plotly_chart(px.bar(df_tdk_grp.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Audio Tendik)'), use_container_width=True)
        with tt_d3:
            st.plotly_chart(px.bar(df_tdk_grp.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Video Tendik)'), use_container_width=True)
        with tt_d4:
            st.plotly_chart(px.bar(df_tdk_grp.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Unit Kerja', orientation='h', text_auto='.1f', title='Top 10 Unit (Total Jam Screen Share Tendik)'), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏆 Top 10 Individu Tendik Teraktif")
        tt_i1, tt_i2, tt_i3, tt_i4 = st.tabs(["📊 Frekuensi Koordinasi", "🎙️ Audio Terlama", "📹 Video Terlama", "💻 Screen Share Terlama"])
        with tt_i1:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Meeting Count').sort_values('Meeting Count'), x='Meeting Count', y='Nama_Tampil', orientation='h', text_auto='.0f', title='Top 10 Tendik: Frekuensi Rapat Terbanyak'), use_container_width=True)
        with tt_i2:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Audio Duration (Jam)').sort_values('Audio Duration (Jam)'), x='Audio Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Durasi Audio (Jam) Terlama'), use_container_width=True)
        with tt_i3:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Video Duration (Jam)').sort_values('Video Duration (Jam)'), x='Video Duration (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Durasi Video (Jam) Terlama'), use_container_width=True)
        with tt_i4:
            st.plotly_chart(px.bar(df_tendik_eda.nlargest(10, 'Screen Share (Jam)').sort_values('Screen Share (Jam)'), x='Screen Share (Jam)', y='Nama_Tampil', orientation='h', text_auto='.1f', title='Top 10 Tendik: Durasi Screen Share (Jam) Terlama'), use_container_width=True)
