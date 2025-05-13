# My13-FashionAI-Lookbook-Generator-
GenAI

Here's a new fashion-related AI project idea with full code, explanation, and steps to run it in VS Code and GitHub:

---

## 👟 FashionAI Lookbook Generator - "Outfit to Lookbook"

### 💡 Project Summary:

This project takes an uploaded fashion outfit photo and automatically generates a virtual *lookbook* page. It extracts outfit details using a vision-language model, recommends similar styled pieces, and generates descriptions, tags, and even a short blog for each look. Think of it as your personal fashion catalog generator!

---

### ✅ Features:

1. **Upload Outfit Image** - User uploads a fashion image.
2. **AI Outfit Description Generator** - Vision-language AI describes outfit.
3. **Auto Tag & Hashtag Generation** - Tags and social-ready hashtags are generated.
4. **Lookbook Page Generator** - Combines everything into a clean digital catalog entry.
5. **Save & Download Lookbook** - Download your generated fashion lookbook as an HTML or image.

---

### 📁 Folder Structure:

```
FashionLookbookAI/
│
├── app.py
├── requirements.txt
├── sample_data/
│   └── outfit1.jpg
└── README.md
```

---

### 🧠 `app.py` – Full Code (Streamlit + Transformers):

```python
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
```

---

### 📦 `requirements.txt`:

```
streamlit
torch
transformers
Pillow
```

---

### 📘 `README.md`

````markdown
# 👗 FashionAI Lookbook Generator

This project lets you upload an outfit photo and automatically generate a digital lookbook entry using AI (BLIP vision-language model).

## ✅ Features

- Upload fashion outfit images.
- Auto-generate outfit description using vision AI.
- Hashtag and tag generator.
- Lookbook blog caption generator.
- Save and reuse entries.

## 🚀 How to Run Locally in VS Code

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/FashionLookbookAI.git
   cd FashionLookbookAI
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Streamlit:

   ```bash
   streamlit run app.py
   ```

4. Upload outfit image and explore results!

## 📌 How to Deploy on GitHub Pages

* This is a Streamlit app; deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) or use [Streamlit Sharing](https://share.streamlit.io/).

---

## ✨ Example

![example](sample_data/outfit1.jpg)

Outfit Description:

> A person wearing a white oversized jacket with tan trousers and sneakers.

---

Made with 💙 using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)

```
