import streamlit as st
import tempfile, os
from PIL import Image
from validator import validate

st.set_page_config(page_title="Photo Compliance Validator", page_icon="🪪", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #1a1a2e; color: #e8e8f0; }

.verdict-pass {
    background: #0f3d2a;
    border: 1.5px solid #34d399;
    border-radius: 10px;
    padding: 16px 22px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #34d399;
    margin: 12px 0 16px 0;
}
.verdict-fail {
    background: #3d1a1a;
    border: 1.5px solid #f87171;
    border-radius: 10px;
    padding: 16px 22px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f87171;
    margin: 12px 0 16px 0;
}
.check-pass {
    background: #1e2d26;
    border: 1px solid #2d4a38;
    border-left: 3px solid #34d399;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    overflow: hidden;
}
.check-fail {
    background: #2d1e1e;
    border: 1px solid #4a2d2d;
    border-left: 3px solid #f87171;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    overflow: hidden;
}
.check-name {
    font-weight: 600;
    font-size: 0.88rem;
    color: #e8e8f0;
    margin-bottom: 3px;
}
.check-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #7a7a9a;
    margin-top: 2px;
}
.check-msg {
    font-size: 0.82rem;
    color: #fca5a5;
    margin: 4px 0 3px 0;
    line-height: 1.4;
}
.reasons-box {
    background: #2d1a1a;
    border: 1px solid #6b3030;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 8px;
}
.reasons-box p {
    font-size: 0.84rem;
    color: #fca5a5;
    margin: 5px 0;
    line-height: 1.5;
}
.label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin: 20px 0 8px 0;
    padding-bottom: 5px;
    border-bottom: 1px solid #2a2a3e;
}
.empty-state {
    text-align: center;
    color: #9090b0;
    padding: 3rem;
    border: 1px dashed #6060aa;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🪪 Photo Compliance Validator")
st.markdown("<p style='font-size:1.1rem; color:#aaa;'>Validate your photo for government & exam applications</p>", unsafe_allow_html=True)
st.divider()

uploaded = st.file_uploader("Upload your photo (JPG only, 20KB – 200KB)", type=["jpg", "jpeg"])

if uploaded:
    size_kb = uploaded.size / 1024
    if size_kb < 20:
        st.error(f"File too small ({size_kb:.1f} KB). Minimum size is 20 KB.")
        st.stop()
    if size_kb > 200:
        st.error(f"File too large ({size_kb:.1f} KB). Maximum size is 200 KB.")
        st.stop()

if not uploaded:
    st.markdown('<div class="empty-state">Upload a photo above to validate it</div>', unsafe_allow_html=True)
    st.stop()

col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    image = Image.open(uploaded)
    st.image(image, width=280)
    st.caption(f"{image.width} × {image.height} px  |  {uploaded.size // 1024} KB")

with col2:
    with st.spinner("Running checks..."):
        ext = os.path.splitext(uploaded.name)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        result = validate(tmp_path)
        os.unlink(tmp_path)

    if result["overall"] == "PASS":
        st.markdown('<div class="verdict-pass">✅ &nbsp; PHOTO COMPLIANT</div>', unsafe_allow_html=True)
    else:
        fail_count = sum(1 for c in result["checks"] if not c["passed"])
        st.markdown(
            f'<div class="verdict-fail">❌ &nbsp; NOT COMPLIANT — {fail_count} issue{"s" if fail_count > 1 else ""} found</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="label">Check Results</div>', unsafe_allow_html=True)
    for c in result["checks"]:
        if c["passed"]:
            st.markdown(f"""
            <div class="check-pass">
                <div class="check-name">✓ &nbsp; {c['name']}</div>
                <div class="check-score">{c['score']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="check-fail">
                <div class="check-name">✗ &nbsp; {c['name']}</div>
                <div class="check-msg">{c['message']}</div>
                <div class="check-score">{c['score']}</div>
            </div>""", unsafe_allow_html=True)

    if result["reasons"]:
        st.markdown('<div class="label">What to fix</div>', unsafe_allow_html=True)
        items = "".join(f'<p>▸ {r}</p>' for r in result["reasons"])
        st.markdown(f'<div class="reasons-box">{items}</div>', unsafe_allow_html=True)
