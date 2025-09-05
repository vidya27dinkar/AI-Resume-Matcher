import streamlit as st
from pathlib import Path
import pandas as pd
from AI_RESUMES_MATCHER_enhanced import generate_ranked_matches
import altair as alt
import re

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="wide",
    page_icon="💼"
)

# ------------------- Background -------------------
def set_background(image_url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{image_url}");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_background("https://images.unsplash.com/photo-1557682250-6e1a1ffae7a0?auto=format&fit=crop&w=1950&q=80")

# ------------------- Helper Functions -------------------
def highlight_keywords(text, keywords):
    for kw in keywords:
        if kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            text = pattern.sub(f"**{kw}**", text)
    return text

def color_score(val):
    if val > 0.75:
        return '#85e085'  # green
    elif val > 0.5:
        return '#ffff99'  # yellow
    else:
        return '#ff9999'  # red

def radar_chart(features: dict, title="Feature Scores"):
    """Draw a radar chart using Altair"""
    df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Score': list(features.values())
    })
    chart = alt.Chart(df).mark_line(point=True).encode(
        theta=alt.Theta("Feature:N", sort=list(features.keys())),
        radius=alt.Radius("Score:Q", scale=alt.Scale(domain=[0,1])),
        color=alt.value("#1f77b4")
    ).properties(width=250, height=250, title=title)
    return chart

# ------------------- Title Section -------------------
st.markdown("""
<div style='background-color: rgba(0,0,0,0.6); padding: 30px; border-radius: 15px; text-align:center; color:white'>
<h1>💼 AI Resume Matcher</h1>
<p>Upload a Job Description and multiple resumes to automatically rank candidates based on skills, education, projects, and experience.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------- Upload Section -------------------
jd_file = st.file_uploader("📄 Upload Job Description", type=["txt","pdf","docx"])
resumes = st.file_uploader("📂 Upload Resumes", type=["txt","pdf","docx"], accept_multiple_files=True)

# ------------------- Sidebar: Matching Options -------------------
st.sidebar.header("⚙️ Matching Options")
weights = {
    'skills': st.sidebar.slider("Skills Weight", 0.0, 1.0, 0.4, 0.05),
    'education': st.sidebar.slider("Education Weight", 0.0, 1.0, 0.2, 0.05),
    'projects': st.sidebar.slider("Projects Weight", 0.0, 1.0, 0.2, 0.05),
    'experience': st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.2, 0.05),
}
top_n = st.sidebar.number_input("Top N Resumes to Display", min_value=1, value=10)

# ------------------- Main Matching -------------------
if st.button("🔍 Match Resumes"):

    if not jd_file:
        st.error("Please upload a Job Description file.")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        jd_path = temp_dir / jd_file.name
        jd_path.write_bytes(jd_file.getbuffer())

        for f in resumes:
            (temp_dir / f.name).write_bytes(f.getbuffer())

        with st.spinner("Matching resumes..."):
            df = generate_ranked_matches(str(temp_dir), str(jd_path), weights=weights)

        if df.empty:
            st.warning("No resumes matched or found.")
        else:
            st.success(f"Found {len(df)} resumes. Showing top {top_n} results.")

            # ------------------- Top Metrics -------------------
            top_scores = df.head(top_n)
            avg_score = top_scores['overall_score'].mean()
            col1, col2 = st.columns(2)
            col1.metric("📈 Average Overall Score", f"{avg_score:.2f}")
            col2.metric("🏆 Top Resume", top_scores.iloc[0]['file'])

            # ------------------- Interactive Bar Chart -------------------
            chart = alt.Chart(top_scores).mark_bar().encode(
                x=alt.X('overall_score', title='Overall Score'),
                y=alt.Y('file', sort='-x', title='Resume'),
                color=alt.Color('overall_score', scale=alt.Scale(scheme='greenyelloworange'))
            )
            st.altair_chart(chart, use_container_width=True)

            # ------------------- Resume Details -------------------
            for idx, row in top_scores.iterrows():
                st.markdown("---")
                st.subheader(f"{row['file']} | Overall Score: <span style='background-color:{color_score(row['overall_score'])};padding:5px;border-radius:5px'>{row['overall_score']:.2f}</span>", unsafe_allow_html=True)
                col1, col2 = st.columns([2,3])

                features = {
                    'Skills': row['skills_match'],
                    'Education': row['education_match'],
                    'Projects': row['projects_match'],
                    'Experience': row['experience_match']
                }

                with col1:
                    st.markdown("<div style='background-color: rgba(255,255,255,0.8); padding:15px; border-radius:10px'>", unsafe_allow_html=True)
                    st.markdown("### 🔹 Feature Breakdown")
                    for f, s in features.items():
                        st.markdown(f"{f}: {s:.2f} " + "▇" * int(s*20))
                    st.altair_chart(radar_chart(features, title=row['file']), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<div style='background-color: rgba(255,255,255,0.8); padding:15px; border-radius:10px'>", unsafe_allow_html=True)
                    st.markdown("### 🔹 Highlighted Resume Details")
                    if row['resume_skills']:
                        st.markdown("**Skills:** " + highlight_keywords(row['resume_skills'], row['resume_skills'].split(',')))
                    if row['resume_education']:
                        st.markdown("**Education:** " + highlight_keywords(row['resume_education'], row['resume_education'].split(';')))
                    if row['resume_projects']:
                        st.markdown("**Projects:** " + highlight_keywords(row['resume_projects'], row['resume_projects'].split(';')))
                    st.markdown(f"**Experience Years:** {row['resume_experience_years']}")
                    st.markdown("</div>", unsafe_allow_html=True)

            # ------------------- Download CSV -------------------
            csv = df.to_csv(index_label='rank').encode('utf-8')
            st.download_button(
                label="📥 Download Ranked CSV",
                data=csv,
                file_name="ranked_resumes.csv",
                mime="text/csv"
            )


