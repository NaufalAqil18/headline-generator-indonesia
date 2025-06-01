import requests
import streamlit as st

API_URL = "http://localhost:8000/generate/"

# Page config
st.set_page_config(
    page_title="Article Title Generator", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark mode, modern card style, and professional look
st.markdown("""
    <style>
    body, .main, .stApp {
        background: #181c2a !important;
        color: #e3e9f7 !important;
    }
    .stTitle {
        color: #ffb86c;
        font-size: 2.7rem !important;
        margin-bottom: 2rem !important;
        font-weight: 900;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #1e293b;
    }
    .custom-header {
        background: linear-gradient(90deg, #232946 0%, #3b3f5c 100%);
        border-radius: 22px;
        box-shadow: 0 8px 32px 0 rgba(30,136,229,0.18);
        padding: 2.2rem 2rem 1.2rem 2rem;
        margin-bottom: 2.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .header-icon {
        font-size: 2.7rem;
        color: #7fd1fc;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 2px 8px #1e88e5);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffb86c;
        letter-spacing: 1.5px;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .header-desc {
        color: #e3e9f7;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #232946 0%, #2b2e4a 100%);
        padding: 1.7rem 1.5rem 1.2rem 1.5rem;
        border-radius: 18px;
        box-shadow: 0 6px 24px 0 rgba(30,136,229,0.18), 0 1.5px 4px 0 rgba(30,136,229,0.10);
        margin: 1.2rem 0 0.7rem 0;
        border: 2px solid #232946;
        transition: box-shadow 0.2s;
    }
    .result-card-gemini {
        border-left: 6px solid #7fd1fc;
    }
    .result-card-bert {
        border-left: 6px solid #43a047;
    }
    .result-card-tuning {
        border-left: 6px solid #ffb86c;
    }
    .model-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
        color: #7fd1fc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .headline-text {
        font-size: 1.18rem;
        font-weight: 600;
        color: #fff;
        background: #232946;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 4px 0 rgba(30,136,229,0.04);
        word-break: break-word;
        border: 1.5px solid #7fd1fc22;
    }
    .metrics-container {
        background-color: #232946;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        color: #7fd1fc;
        font-size: 1rem;
        border: 1px solid #2b2e4a;
    }
    .stButton button {
        width: 100%;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        background-color: #7fd1fc;
        color: #232946;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1e88e5;
        color: #fff;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(30,136,229,0.18);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #232946;
        padding: 1rem;
        font-size: 1rem;
        background: #181c2a;
        color: #fff;
    }
    .stTextArea textarea:focus {
        border-color: #7fd1fc;
        box-shadow: 0 0 0 2px #7fd1fc44;
    }
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class='custom-header'>
    <div class='header-icon'>🎤</div>
    <div class='header-title'>Article Title Generator</div>
    <div class='header-desc'>Buat judul artikel menarik dengan AI: Gemini, BERT, dan Tuning Model</div>
</div>
""", unsafe_allow_html=True)

# Info section with tabs
tab1, tab2 = st.tabs(["About", "How to Use"])
with tab1:
    st.markdown("""
        This tool uses two powerful AI models to generate engaging titles for your articles:
        - 🤖 **Gemini AI**: Google's advanced language model
        - 🔬 **BERT**: Specialized Indonesian summarization model
    """)
with tab2:
    st.markdown("""
        1. Paste your article content in the text area below
        2. Click "Generate Titles" button
        3. Compare results from both AI models
        4. Choose the title that best fits your needs
    """)

# Main content section
st.markdown("### 📄 Your Article")
article_content = st.text_area(
    "",  # Empty label since we use markdown above
    height=300,
    placeholder="Paste your article content here (minimum 50 characters)...",
    help="For best results, paste at least one full paragraph of your article"
)

# Progress bar placeholder
progress_placeholder = st.empty()

# generate titles
if st.button("🎯 Generate Titles", use_container_width=True):
    if not article_content:
        st.error("⚠️ Please enter your article content first.")
    elif len(article_content.strip()) < 50:
        st.warning("⚠️ Article content is too short. Please provide at least 50 characters.")
    else:
        try:
            # Show progress
            with st.spinner("🔄 AI models are working their magic..."):
                # Prepare payload
                payload = {"content": article_content}
                
                # Send request to API
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            
            st.success("✨ Titles generated successfully!")
            
            # Results section
            st.markdown("### <span style='color:#7fd1fc;font-weight:800;'>Generated Titles</span>", unsafe_allow_html=True)
            
            # Display results vertically, one card per model, with color accent
            def show_result_card(title, subtitle, metrics, icon, card_class):
                st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
                st.markdown(f"<div class='model-title'>{icon} {subtitle}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='headline-text'>{title}</div>", unsafe_allow_html=True)
                if metrics:
                    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                    st.markdown(f"""
                        <span style='color:#7fd1fc'>📊 <b>Metrics</b>:</span><br>
                        • <b>Words</b>: {metrics['length']}<br>
                        • <b>Characters</b>: {metrics['character_count']}
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            show_result_card(
                data.get('gemini_title', '– No output –'),
                "Gemini AI",
                data.get('gemini_metrics'),
                "🤖",
                "result-card-gemini"
            )
            show_result_card(
                data.get('bert_title', '– No output –'),
                "BERT Model",
                data.get('bert_metrics'),
                "🔬",
                "result-card-bert"
            )
            show_result_card(
                data.get('tuning_title', '– No output –'),
                "Tuning Model",
                data.get('tuning_metrics'),
                "🧪",
                "result-card-tuning"
            )
            
            # Comparison tip
            st.info("💡 **Tip**: Compare both titles and choose the one that best matches your article's tone and purpose!")
                
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: Make sure the backend server is running! ({str(e)})")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Streamlit, FastAPI, Gemini, and BERT
    </div>
""", unsafe_allow_html=True)