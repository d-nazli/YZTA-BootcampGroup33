import streamlit as st

# 🎯 Page settings
st.set_page_config(
    page_title="MoodSenseAI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🌄 Background image and styles
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    visibility: hidden;
}}

textarea, .stTextInput > div > div > input {{
    background-color: rgba(255, 255, 255, 0.8);
    color: #333;
    border-radius: 10px;
    padding: 10px;
}}

.stButton > button {{
    background-color: #6ab04c;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}}

.stMarkdown h1, .stMarkdown h2 {{
    color: #ffffff;
    text-shadow: 0px 0px 10px rgba(0,0,0,0.7);
    text-align: center;
}}

.stMarkdown p {{
    color: #ffffff;
    text-shadow: 0px 0px 5px rgba(0,0,0,0.6);
    text-align: center;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🧠 Title and description
st.markdown("<h1>MoodSenseAI</h1>", unsafe_allow_html=True)
st.markdown("<p>An AI-powered support system designed to understand emotional states through natural language input.</p>", unsafe_allow_html=True)
st.markdown("")

# ✍️ Text input area
user_input = st.text_area("✍️ Share your thoughts:", placeholder="Feel free to express how you're feeling today...", height=180)

# 🔍 Analyze button and placeholder
if st.button("🔍 Analyze Mood"):
    if user_input.strip() == "":
        st.warning("Please enter something first.")
    else:
        st.info("🧠 AI model is not yet integrated in this demo. Coming soon!")
        st.success("💬 Your entry has been received. You're not alone — we're here for you.")

# 👣 Footer
st.markdown("""
<br><hr>
<div style='text-align: center; color: white; font-size: 0.85em; text-shadow: 0 0 5px black;'>
© 2025 MoodSenseAI – Under Development
</div>
""", unsafe_allow_html=True)
