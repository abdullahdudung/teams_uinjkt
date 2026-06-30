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

# Tema warna kustom universitas
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
    
    def proses_metrik_dan_resensi(df):
        # 1. Konversi Durasi dari Detik ke Menit
        df['Audio Duration (Menit)'] = df['Audio Duration In Seconds'] / 60
        df['Video Duration (Menit)'] = df['Video Duration In Seconds'] / 60
        df['Screen Share (Menit)'] = df['Screen Share Duration In Seconds'] / 60
        df['Total_Duration (Menit)'] = (
            df['Audio Duration (Menit)'] + 
            df['Video Duration (Menit)'] + 
            df['Screen Share (Menit)']
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
                
        df['Tingkat_Aktivitas_Resensi'] = df.apply(golongkan_aktivitas, axis=1)
        
        # Penanganan Nama untuk Papan Peringkat Individu
        if 'Nama' in df.columns:
            df['Nama_Tampil'] = df['Nama'].fillna(df['Username'])
        else:
            df['Nama_Tampil'] = df['Username']
            
        return df

    # Proses masing-masing dataframe tanpa menghapus data kosong agar retensi terbaca sempurna
    df_mhs = proses_metrik_dan_resensi(df_mhs)
    df_mhs.insert(0, 'User ID', ['MHS_' + str(i).zfill(5) for i in range(1, len(df_mhs) + 1)])
    
    df_staff['Role'] = df_staff['Role'].fillna('Tendik').replace({'tendik': 'Tendik', 'dosen': 'Dosen'})
    df_staff = proses_metrik_dan_resensi(df_staff)
    df_staff.insert(0, 'User ID', ['STAFF_' + str(i).zfill(4) for i in range(1, len(df_staff) + 1)])
    
    return df_mhs, df_staff

# ==========================================
# PEMBACAAN DATA UTAMA
# ==========================================
df_mhs, df_staff = load_and_preprocess_data()

if df_mhs is None or df_staff is None:
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.error("⚠️ File `mhs juni.csv` dan/atau `staff juni.csv` tidak ditemukan di direktori aplikasi.")
else:
    # Header Aplikasi Utama
    st.title("📊 Analisis Aktivitas Penggunaan Microsoft Teams")
    st.markdown("### Laporan Periode Akumulasi 180 Hari (6 Bulan) | UIN Syarif Hidayatullah Jakarta")
    st.markdown("---")
    
    # Navigasi Tiga Tab Utama
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
            <h2 style="color:white;margin:0">📝 RINGKASAN EKSEKUTIF KESELURUHAN</h2>
            <p style="color:#E0F2F1;margin:5px 0 0 0">Gambaran Makro Adopsi Lisensi Microsoft 365 Kampus Selama Periode 180 Hari</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrik Utama
        col1, col2, col3 = st.columns(3)
        total_mhs_lisensi = len(df_mhs)
        total_dosen_lisensi = len(df_staff[df_staff['Role'] == 'Dosen'])
        total_tendik_lisensi = len(df_staff[df_staff['Role'] == 'Tendik'])
        
        col1.metric("Total Lisensi Mahasiswa Terdata", f"{total_mhs_lisensi:,}".replace(',', '.'))
        col2.metric("Total Lisensi Dosen Terdata", f"{total_dosen_lisensi:,}".replace(',', '.'))
        col3.metric("Total Lisensi Tendik Terdata", f"{total_tendik_lisensi:,}".replace(',', '.'))
        
        st.markdown("---")
        
        # Grafik Perbandingan Makro Kontribusi Durasi
        st.markdown("### 📊 Profil Rata-rata Beban Interaksi Digital")
        col_prof1, col_prof2 = st.columns(2)
        
        with col_prof1:
            st.markdown("#### 🎓 Rata-rata Penggunaan Fitur oleh Mahasiswa")
            st.info(f"🎙️ Total Audio: **{df_mhs['Audio Duration (Menit)'].mean():.1f}** Menit / User\n\n📹 Total Video: **{df_mhs['Video Duration (Menit)'].mean():.1f}** Menit / User\n\n💻 Total Berbagi Layar: **{df_mhs['Screen Share (Menit)'].mean():.1f}** Menit / User")
            st.caption("**Interpretasi Komparatif:** Aktivitas mahasiswa didominasi secara mutlak oleh komunikasi suara (*Audio*) dan visualisasi materi (*Screen Share*). Hal ini mencerminkan karakteristik kelas *hybrid* yang bersifat asimetris di mana mahasiswa lebih banyak memosisikan diri sebagai audiens penerima informasi.")
            
        with col_prof2:
            st.markdown("#### 💼 Rata-rata Penggunaan Fitur oleh Dosen & Tendik")
            st.success(f"🎙️ Total Audio: **{df_staff['Audio Duration (Menit)'].mean():.1f}** Menit / User\n\n📹 Total Video: **{df_staff['Video Duration (Menit)'].mean():.1f}** Menit / User\n\n💻 Total Berbagi Layar: **{df_staff['Screen Share (Menit)'].mean():.1f}** Menit / User")
            st.caption("**Interpretasi Komparatif:** Profil *Staff* menunjukkan intensitas durasi interaksi yang jauh lebih tinggi di semua sektor fitur dibandingkan mahasiswa. Hal ini wajar mengingat kedudukan Dosen dan Tendik sebagai fasilitator utama pertemuan kelas, rapat koordinasi tingkat birokrasi, dan penyelenggara layanan administratif harian.")

        st.markdown("---")
        st.markdown("### 📋 Pratonton Sampel Data (Identitas Disamarkan)")
        col_pre1, col_pre2 = st.columns(2)
        with col_pre1:
            st.markdown("**Sampel Log Mahasiswa**")
            st.dataframe(df_mhs[['User ID', 'Fakultas', 'Prodi', 'Meeting Count', 'Total_Duration (Menit)', 'Tingkat_Aktivitas_Recency']].head(5), use_container_width=True, hide_index=True)
        with col_pre2:
            st.markdown("**Sampel Log Dosen & Tendik**")
            st.dataframe(df_staff[['User ID', 'Role', 'Unit Kerja', 'Meeting Count', 'Total_Duration (Menit)', 'Tingkat_Aktivitas_Recency']].head(5), use_container_width=True, hide_index=True)

    # ----------------------------------------
    # TAB 2: EDA MAHASISWA
    # ----------------------------------------
    with tab2:
        st.markdown("""
        <div style="background-color:#1E88E5;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">🎓 EXPLORATORY DATA ANALYSIS (MAHASISWA)</h2>
            <p style="color:#E3F2FD;margin:5px 0 0 0">Analisis Perilaku Digital, Retensi Resensi Akses, dan Pemetaan Sektoral Fakultas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. ANALISIS RESENSI AKSES TERAKHIR (TINGKAT AKTIVITAS)
        st.markdown("### 🔄 Distribusi Tingkat Aktivitas Berdasarkan Akses Terakhir (Resensi)")
        
        status_order = ["Sangat Aktif (Akses 0-7 Hari Lalu)", "Aktif (Akses 8-30 Hari Lalu)", "Cukup Aktif (Akses 31-90 Hari Lalu)", "Pasif (Akses >90 Hari Lalu)", "Tidak Aktif (Dalam 180 Hari)"]
        mhs_status = df_mhs['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
        mhs_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah Pengguna']
        
        fig_mhs_rec = px.bar(mhs_status, x='Tingkat Aktivitas Resensi', y='Jumlah Pengguna', text='Jumlah Pengguna',
                             color='Tingkat Aktivitas Resensi', color_discrete_sequence=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'])
        fig_mhs_rec.update_layout(showlegend=False)
        st.plotly_chart(fig_mhs_rec, use_container_width=True)
        
        with st.expander("💡 Lihat Interpretasi & Kesimpulan Analisis Retensi Mahasiswa", expanded=True):
            st.markdown("""
            * **Interpretasi Grafik:** Grafik resensi di atas membagi total populasi mahasiswa ke dalam 5 spektrum aktivitas berdasarkan seberapa dekat jarak hari akses terakhir mereka dari tanggal pembaruan sistem. Kelompok *Sangat Aktif* dan *Aktif* menggambarkan porsi mahasiswa yang terikat langsung dalam proses pembelajaran digital kontemporer.
            * **Kesimpulan Strategis:** Volume pengguna yang menumpuk pada klaster *Tidak Aktif (Dalam 180 Hari)* memberikan indikasi awal penting bagi manajemen IT. Kelompok ini mewakili mahasiswa yang memiliki akun resmi namun tidak menyentuh *platform* MS Teams sama sekali selama satu semester penuh. Manajemen dapat memanfaatkan data ini untuk melakukan efisiensi atau pembersihan alokasi lisensi yang tidak produktif.
            """)
            
        st.markdown("---")
        
        # 2. PERINGKAT TOP 10 FAKULTAS TERGERAK (AGREGAT)
        st.markdown("### 🏆 Top 10 Fakultas Teratas Berdasarkan Metrik Penggunaan (Agregat Periode 180 Hari)")
        
        df_fakultas_grp = df_mhs.groupby('Fakultas')[['Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)']].sum().reset_index()
        
        tab_m_meet, tab_m_aud, tab_m_vid, tab_m_scr = st.tabs([
            "📊 Total Pertemuan Rapat", 
            "🎙️ Total Durasi Audio", 
            "📹 Total Durasi Video", 
            "💻 Total Durasi Berbagi Layar"
        ])
        
        with tab_m_meet:
            top_10 = df_fakultas_grp.nlargest(10, 'Meeting Count').sort_values('Meeting Count', ascending=True)
            st.plotly_chart(px.bar(top_10, x='Meeting Count', y='Fakultas', orientation='h', text_auto='.0f', title='Top 10 Fakultas: Akumulasi Frekuensi Kelas Virtual', color_discrete_sequence=['#004D40']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Sektor ini menampilkan total intensitas pembukaan kelas virtual daring. Fakultas peringkat teratas menunjukkan kesiapan implementasi kurikulum berbasis integrasi teknologi informasi (*Smart Campus*) tertinggi di lingkungan mahasiswa.")
            
        with tab_m_aud:
            top_10 = df_fakultas_grp.nlargest(10, 'Audio Duration (Menit)').sort_values('Audio Duration (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10, x='Audio Duration (Menit)', y='Fakultas', orientation='h', text_auto='.0f', title='Top 10 Fakultas: Akumulasi Durasi Audio (Menit)', color_discrete_sequence=['#1E88E5']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Peringkat durasi audio mencerminkan total beban diskusi verbal harian. Fakultas yang memimpin di sektor ini mengindikasikan adanya budaya interaksi kelas interaktif berbasis suara yang masif sepanjang periode studi berlangsung.")
            
        with tab_m_vid:
            top_10 = df_fakultas_grp.nlargest(10, 'Video Duration (Menit)').sort_values('Video Duration (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10, x='Video Duration (Menit)', y='Fakultas', orientation='h', text_auto='.0f', title='Top 10 Fakultas: Akumulasi Durasi Kamera Menyala (Menit)', color_discrete_sequence=['#D81B60']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Durasi video mengukur kepatuhan dan formalitas interaksi visual tatap muka virtual. Tingginya angka di fakultas tertentu menandakan adanya kebijakan penegakan kedisiplinan menyalakan kamera (*on-cam*) yang berjalan efektif.")
            
        with tab_m_scr:
            top_10 = df_fakultas_grp.nlargest(10, 'Screen Share (Menit)').sort_values('Screen Share (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10, x='Screen Share (Menit)', y='Fakultas', orientation='h', text_auto='.0f', title='Top 10 Fakultas: Akumulasi Durasi Presentasi Layar (Menit)', color_discrete_sequence=['#FFC107']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Grafik *Screen Share* ini menangkap indikator pemaparan materi kerja digital (seperti pengerjaan tugas kelompok, seminar, atau bedah studi kasus ilmiah) yang digerakkan langsung oleh inisiatif mahasiswa.")

    # ----------------------------------------
    # TAB 3: EDA DOSEN DAN TENDIK
    # ----------------------------------------
    with tab3:
        st.markdown("""
        <div style="background-color:#FF8F00;padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0">💼 EXPLORATORY DATA ANALYSIS (DOSEN & TENDIK)</h2>
            <p style="color:#FFF8E1;margin:5px 0 0 0">Analisis Pola Komunikasi Birokrasi, Resensi Akses Instansi, dan Distribusi Beban Kerja Sektoral Unit Kerja</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. ANALISIS RESENSI AKSES TERAKHIR STAFF (TINGKAT AKTIVITAS)
        st.markdown("### 🔄 Distribusi Tingkat Aktivitas Berdasarkan Akses Terakhir (Resensi Staff)")
        
        staff_status = df_staff['Tingkat_Aktivitas_Recency'].value_counts().reindex(status_order, fill_value=0).reset_index()
        staff_status.columns = ['Tingkat Aktivitas Resensi', 'Jumlah Pengguna']
        
        fig_stf_rec = px.bar(staff_status, x='Tingkat Aktivitas Resensi', y='Jumlah Pengguna', text='Jumlah Pengguna',
                             color='Tingkat Aktivitas Resensi', color_discrete_sequence=['#004D40', '#00796B', '#80CBC4', '#FFB74D', '#E53935'])
        fig_stf_rec.update_layout(showlegend=False)
        st.plotly_chart(fig_stf_rec, use_container_width=True)
        
        with st.expander("💡 Lihat Interpretasi & Kesimpulan Analisis Retensi Staff / Pegawai", expanded=True):
            st.markdown("""
            * **Interpretasi Grafik:** Visualisasi ini memetakan konsistensi dosen dan tenaga kependidikan dalam memanfaatkan ruang kolaborasi korporat institusi. Berbeda dengan mahasiswa, kelompok pegawai idealnya terkonsentrasi penuh pada zona *Sangat Aktif* dan *Aktif* untuk menjaga kelancaran koordinasi administrasi internal kampus.
            * **Kesimpulan Strategis:** Jika ditemukan lonjakan angka pegawai pada kategori *Pasif* atau *Tidak Aktif*, hal tersebut mengindikasikan adanya hambatan adopsi teknologi birokrasi digital atau kecenderungan unit kerja terkait yang kembali beralih menggunakan aplikasi komunikasi non-resmi instansi (seperti WhatsApp grup personal). Ini dapat menjadi acuan bagi bagian kepegawaian untuk menyelenggarakan pelatihan tata kerja digital lanjutan.
            """)
            
        st.markdown("---")
        
        # 2. PERINGKAT TOP 10 UNIT KERJA TERGERAK (AGREGAT)
        st.markdown("### 🏆 Top 10 Unit Kerja Teratas Berdasarkan Metrik Penggunaan (Agregat Periode 180 Hari)")
        
        df_unit_grp = df_staff.groupby('Unit Kerja')[['Meeting Count', 'Audio Duration (Menit)', 'Video Duration (Menit)', 'Screen Share (Menit)']].sum().reset_index()
        
        tab_s_meet, tab_s_aud, tab_s_vid, tab_s_scr = st.tabs([
            "📊 Total Pertemuan Rapat", 
            "🎙️ Total Durasi Audio", 
            "📹 Total Durasi Video", 
            "💻 Total Durasi Berbagi Layar"
        ])
        
        with tab_s_meet:
            top_10_u = df_unit_grp.nlargest(10, 'Meeting Count').sort_values('Meeting Count', ascending=True)
            st.plotly_chart(px.bar(top_10_u, x='Meeting Count', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit Kerja: Akumulasi Frekuensi Koordinasi Virtual', color_discrete_sequence=['#00796B']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Data ini menyoroti Unit Kerja (Fakultas/Lembaga/Pusat) yang paling intensif menginisiasi pertemuan dinas secara daring. Peringkat teratas mendefinisikan unit yang memiliki dinamika koordinasi manajerial berbasis virtual paling progresif.")
            
        with tab_s_aud:
            top_10_u = df_unit_grp.nlargest(10, 'Audio Duration (Menit)').sort_values('Audio Duration (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10_u, x='Audio Duration (Menit)', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit Kerja: Akumulasi Durasi Audio (Menit)', color_discrete_sequence=['#1565C0']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Total akumulasi durasi audio mengindikasikan waktu yang dihabiskan para pimpinan, dosen, dan staf administrasi unit kerja dalam berdiskusi merumuskan kebijakan operasional di ruang rapat virtual.")
            
        with tab_s_vid:
            top_10_u = df_unit_grp.nlargest(10, 'Video Duration (Menit)').sort_values('Video Duration (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10_u, x='Video Duration (Menit)', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit Kerja: Akumulasi Durasi Kamera Menyala (Menit)', color_discrete_sequence=['#C2185B']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Durasi interaksi visual mencerminkan transparansi dan keterikatan profesional pegawai dalam bekerja secara *remote* atau *hybrid*. Unit kerja dengan peringkat tinggi menunjukkan etos kedisiplinan rapat virtual yang sangat baik.")
            
        with tab_s_scr:
            top_10_u = df_unit_grp.nlargest(10, 'Screen Share (Menit)').sort_values('Screen Share (Menit)', ascending=True)
            st.plotly_chart(px.bar(top_10_u, x='Screen Share (Menit)', y='Unit Kerja', orientation='h', text_auto='.0f', title='Top 10 Unit Kerja: Akumulasi Durasi Presentasi Layar (Menit)', color_discrete_sequence=['#F57C00']), use_container_width=True)
            st.info("**Interpretasi & Kesimpulan:** Peringkat presentasi layar memetakan tingkat intensitas pemaparan dokumen dinas, sistem aplikasi pangkalan data, laporan keuangan, atau materi kurikulum akademik ilmiah yang dipaparkan secara langsung oleh aparatur unit kerja.")
