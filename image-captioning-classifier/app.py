import streamlit as st
from PIL import Image
from src.captioner import ImageCaptioner

st.set_page_config(page_title="Image Captioning Classifier", page_icon="🖼️")
st.title("🖼️ Image Captioning Classifier")
st.write("Generate a natural-language caption for an uploaded image using a CNN-LSTM model.")

@st.cache_resource
def load_model():
    return ImageCaptioner()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Generate caption", type="primary"):
        try:
            model = load_model()
            with st.spinner("Generating caption..."):
                caption = model.predict(image)
            st.success(caption)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Caption generation failed: {exc}")
            st.info("Check that the trained model files and dependencies are installed.")
