import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Dashboard Microsoft Teams UIN Jakarta",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📊 Dashboard Microsoft Teams")
st.subheader("UIN Syarif Hidayatullah Jakarta")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "dataset_teams_uinjkt.xlsx"
    )

    return df

try:
    df = load_data()

except:

    st.error(
        "File dataset_teams_uinjkt.xlsx tidak ditemukan"
    )

    st.stop()

# =====================================================
# CLEANING
# =====================================================

df["Role"] = df["Role"].fillna(
    df["Role"].mode()[0]
)

df["Role"] = df["Role"].replace({
    "tendik":"Tendik"
})

df = df.dropna(
    subset=["Last Activity Date"]
)

df["Total_Duration"] = (

    df["Audio Duration In Seconds"]

    + df["Video Duration In Seconds"]

    + df["Screen Share Duration In Seconds"]

)

# =====================================================
# ACTIVITY LEVEL
# =====================================================

def kategori(x):

    if x <= 5:
        return "Rendah"

    elif x <= 15:
        return "Sedang"

    else:
        return "Tinggi"

df["Activity_Level"] = (
    df["Meeting Count"]
    .apply(kategori)
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Menu")

menu = st.sidebar.radio(
    "Pilih Dashboard",
    [
        "📊 Executive Dashboard",
        "📈 Aktivitas Teams",
        "📊 Analisis Lanjutan",
        "🏆 Ranking Pengguna",
        "👨‍🏫 Analisis Role",
        "🤖 Prediksi Aktivitas"
    ]
)

role_filter = st.sidebar.multiselect(
    "Filter Role",
    options=df["Role"].unique(),
    default=df["Role"].unique()
)

df = df[
    df["Role"].isin(role_filter)
]

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

if menu == "📊 Executive Dashboard":

    st.title("📊 Executive Dashboard")

    total_user = len(df)

    total_meeting = int(
        df["Meeting Count"].sum()
    )

    total_audio = round(
        df["Audio Duration In Seconds"].sum()/3600,
        1
    )

    total_video = round(
        df["Video Duration In Seconds"].sum()/3600,
        1
    )

    total_screen = round(
        df["Screen Share Duration In Seconds"].sum()/3600,
        1
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Pengguna",
        total_user
    )

    c2.metric(
        "Meeting",
        total_meeting
    )

    c3.metric(
        "Audio (Jam)",
        total_audio
    )

    c4.metric(
        "Video (Jam)",
        total_video
    )

    c5.metric(
        "Screen Share (Jam)",
        total_screen
    )

    st.divider()

    col1,col2 = st.columns(2)

    role_count = df["Role"].value_counts()

    fig_role = px.pie(
        values=role_count.values,
        names=role_count.index,
        hole=.55,
        title="Distribusi Pengguna"
    )

    col1.plotly_chart(
        fig_role,
        use_container_width=True
    )

    act = (
        df["Activity_Level"]
        .value_counts()
    )

    fig_act = px.bar(
        x=act.index,
        y=act.values,
        color=act.index,
        title="Distribusi Tingkat Aktivitas"
    )

    col2.plotly_chart(
        fig_act,
        use_container_width=True
    )

    st.markdown("## 📌 Insight Otomatis")

    st.info(f"""
    Total pengguna Teams : {total_user}

    Total meeting : {total_meeting}

    Rata-rata meeting per pengguna :
    {df['Meeting Count'].mean():.2f}

    Rata-rata audio :
    {df['Audio Duration In Seconds'].mean()/3600:.2f} jam

    Rata-rata video :
    {df['Video Duration In Seconds'].mean()/3600:.2f} jam
    """)

# =====================================================
# AKTIVITAS TEAMS
# =====================================================

elif menu == "📈 Aktivitas Teams":

    st.title("📈 Aktivitas Teams")

    fig_hist = px.histogram(
        df,
        x="Meeting Count",
        nbins=30,
        title="Distribusi Meeting Count"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    fig_scatter = px.scatter(
        df,
        x="Audio Duration In Seconds",
        y="Video Duration In Seconds",
        color="Meeting Count",
        title="Hubungan Audio dan Video"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

# =====================================================
# ANALISIS LANJUTAN
# =====================================================

elif menu == "📊 Analisis Lanjutan":

    st.title("📊 Analisis Lanjutan")

    corr_cols = [

        "Meeting Count",

        "Audio Duration In Seconds",

        "Video Duration In Seconds",

        "Screen Share Duration In Seconds",

        "Total_Duration"

    ]

    corr = df[corr_cols].corr()

    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Heatmap Korelasi"
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True
    )

    st.dataframe(
        df[corr_cols].describe()
    )

# =====================================================
# RANKING PENGGUNA
# =====================================================

elif menu == "🏆 Ranking Pengguna":

    st.title("🏆 Ranking Pengguna")

    metric = st.selectbox(
        "Pilih Ranking",
        [
            "Meeting Count",
            "Audio Duration In Seconds",
            "Video Duration In Seconds",
            "Screen Share Duration In Seconds"
        ]
    )

    top = (

        df

        .sort_values(
            metric,
            ascending=False
        )

        .head(20)

    )

    fig = px.bar(
        top,
        x=metric,
        y="Username",
        orientation="h",
        color=metric,
        title=f"Top 20 {metric}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(top)

# =====================================================
# ANALISIS ROLE
# =====================================================

elif menu == "👨‍🏫 Analisis Role":

    st.title("👨‍🏫 Analisis Role")

    summary = (

        df

        .groupby("Role")

        .agg({

            "Meeting Count":"mean",

            "Audio Duration In Seconds":"mean",

            "Video Duration In Seconds":"mean",

            "Screen Share Duration In Seconds":"mean"

        })

        .reset_index()

    )

    metric = st.selectbox(
        "Pilih Metrik",
        summary.columns[1:]
    )

    fig = px.bar(
        summary,
        x="Role",
        y=metric,
        color="Role",
        title=f"Perbandingan {metric}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(summary)

# =====================================================
# PREDIKSI AKTIVITAS
# =====================================================

elif menu == "🤖 Prediksi Aktivitas":

    st.title("🤖 Prediksi Aktivitas")

    fitur = [

        "Audio Duration In Seconds",

        "Video Duration In Seconds",

        "Screen Share Duration In Seconds",

        "Total_Duration"

    ]

    X = df[fitur]

    y = df["Activity_Level"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_scaled,
        y_encoded
    )

    audio = st.number_input(
        "Audio Duration",
        value=50000
    )

    video = st.number_input(
        "Video Duration",
        value=45000
    )

    screen = st.number_input(
        "Screen Share Duration",
        value=30000
    )

    if st.button("Prediksi"):

        total = audio + video + screen

        data_baru = pd.DataFrame({

            "Audio Duration In Seconds":[audio],

            "Video Duration In Seconds":[video],

            "Screen Share Duration In Seconds":[screen],

            "Total_Duration":[total]

        })

        data_scaled = scaler.transform(
            data_baru
        )

        pred = model.predict(
            data_scaled
        )[0]

        hasil = encoder.inverse_transform(
            [pred]
        )[0]

        st.success(
            f"Tingkat Aktivitas : {hasil}"
        )

        prob = model.predict_proba(
            data_scaled
        )[0]

        prob_df = pd.DataFrame({

            "Kategori":encoder.classes_,

            "Probabilitas":prob

        })

        fig = px.bar(
            prob_df,
            x="Kategori",
            y="Probabilitas",
            color="Kategori",
            title="Probabilitas Prediksi"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
