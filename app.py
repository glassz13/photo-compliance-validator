import streamlit as st
import tempfile, os
from PIL import Image
from validator import load_model, predict

st.set_page_config(page_title="Face Validator", page_icon="🪪", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: #0f0f17; color: #e0e0f0; }

.card {
    background: #1a1a2e;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.card-pass { border-left: 3px solid #34d399; }
.card-fail { border-left: 3px solid #f87171; }
.card-info { border-left: 3px solid #60a5fa; }

.card-left { display: flex; flex-direction: column; gap: 3px; }
.card-label { font-size: 0.82rem; color: #8888aa; font-weight: 500; }
.card-value { font-size: 0.95rem; color: #e0e0f0; font-weight: 600; }
.card-msg   { font-size: 0.78rem; color: #f87171; margin-top: 2px; }
.card-badge-pass { background: #0d2e20; color: #34d399; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600; }
.card-badge-fail { background: #2e0d0d; color: #f87171; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600; }
.card-badge-info { background: #0d1e3e; color: #60a5fa; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600; }

.verdict-pass {
    background: #0d2e20; border: 1px solid #34d399;
    border-radius: 10px; padding: 14px;
    text-align: center; color: #34d399;
    font-size: 1rem; font-weight: 600; margin: 12px 0;
}
.verdict-fail {
    background: #2e0d0d; border: 1px solid #f87171;
    border-radius: 10px; padding: 14px;
    text-align: center; color: #f87171;
    font-size: 1rem; font-weight: 600; margin: 12px 0;
}
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model("face_attributes.pt")

model = get_model()

st.markdown("## 🪪 Face Compliance Validator")
st.markdown("<p style='color:#8888aa; font-size:0.95rem;'>Upload a photo to check ID/passport photo compliance</p>", unsafe_allow_html=True)
st.divider()

uploaded = st.file_uploader("Upload photo (JPG)", type=["jpg", "jpeg"])

if not uploaded:
    st.markdown("<div style='text-align:center; color:#6060aa; padding:3rem; border:1px dashed #3a3a5a; border-radius:12px; margin-top:1rem;'>Upload a photo to begin</div>", unsafe_allow_html=True)
    st.stop()

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    image = Image.open(uploaded)
    st.image(image, width=240)
    st.caption(str(image.width) + " × " + str(image.height) + " px")

with col2:
    with st.spinner("Analysing..."):
        ext = os.path.splitext(uploaded.name)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        results = predict(model, tmp_path)
        os.unlink(tmp_path)

    failures = [v for v in results.values() if v["passed"] is False]
    if not failures:
        st.markdown('<div class="verdict-pass">✅ &nbsp; PHOTO COMPLIANT</div>', unsafe_allow_html=True)
    else:
        count    = str(len(failures))
        suffix   = "s" if len(failures) > 1 else ""
        st.markdown(
            '<div class="verdict-fail">❌ &nbsp; NOT COMPLIANT — ' + count + ' issue' + suffix + ' found</div>',
            unsafe_allow_html=True
        )

    for attr, res in results.items():
        if res["passed"] is None:
            css   = "card card-info"
            badge = '<span class="card-badge-info">' + res["value"] + '</span>'
        elif res["passed"]:
            css   = "card card-pass"
            badge = '<span class="card-badge-pass">✓ Pass</span>'
        else:
            css   = "card card-fail"
            badge = '<span class="card-badge-fail">✗ Fail</span>'

        msg_html = '<div class="card-msg">' + res["msg"] + '</div>' if res["msg"] else ""

        html = (
            '<div class="' + css + '">'
            '<div class="card-left">'
            '<div class="card-label">' + res["label"] + '</div>'
            '<div class="card-value">' + res["value"] + '</div>'
            + msg_html +
            '</div>'
            + badge +
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
