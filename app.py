import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# 🟡 Must be first Streamlit command
st.set_page_config(page_title="🧾 FashionAI Lookbook Generator", layout="wide")

st.title("👗 Outfit to Lookbook - FashionAI 📖")

# Load BLIP model (vision-to-text)
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

uploaded_file = st.file_uploader("📸 Upload an Outfit Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Outfit", use_column_width=True)

    with st.spinner("🧠 Generating AI Fashion Description..."):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        description = processor.decode(out[0], skip_special_tokens=True)

    st.markdown(f"### 📝 Outfit Description")
    st.success(description)

    # Hashtag Generator
    hashtags = "#" + " #".join(description.lower().replace(".", "").split())
    st.markdown("### 🏷️ Hashtags & Tags")
    st.text_area("Auto-generated Tags", hashtags, height=100)

    # Short Lookbook Blog Post
    blog_post = f"""
    ✨ **Featured Look:** {description}

    This outfit combines elegance and comfort. Perfect for daily wear or casual outings, it adds a stylish flair to your wardrobe. Pair it with neutral accessories and you're good to go!

    **Tags:** {hashtags}
    """

    st.markdown("### 📰 Lookbook Page Entry")
    st.text_area("🧾 Auto-generated Blog Entry", blog_post.strip(), height=200)

    if st.button("💾 Save Lookbook Entry"):
        with open("lookbook_entry.txt", "w", encoding="utf-8") as f:
            f.write(blog_post)
        st.success("✅ Entry saved as 'lookbook_entry.txt'.")
