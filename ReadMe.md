# **Fine-Tuning LLaMA 3.2-11B-Vision for Product Descriptions**  

This project fine-tunes **LLaMA 3.2-11B-Vision** using **LoRA (Low-Rank Adaptation)** to generate **concise, SEO-friendly product descriptions** for kid's clothing images.  

## Blog post
[Read the blog post for more details.](https://ramintoosi.ir/posts/2025/02/blog-post-1/)

---


<p align="center">
  <a href="https://colab.research.google.com/drive/1H9tqCdNrEDO_VqkOC8yLhVX7H0zSA41g?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>


## **📌 Project Overview**  
LLaMA 3.2-11B-Vision is a powerful multimodal model capable of analyzing images. However, its default outputs are too detailed for e-commerce use. This project fine-tunes the model to:  
✅ **Generate short, SEO-optimized product descriptions**  
✅ **Match the branding and marketing style**  
✅ **Efficiently run on consumer-grade GPUs using 4-bit quantization and LoRA**  



## **🛠️ Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/ramintoosi/product_description
cd product_description
```

Follow the [unsloth installation guide](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions).

### **3️⃣ Download & Prepare Dataset**  
The dataset was collected by crawling a web store (publicly available), but the crawling script is not included.  

```python
pip install gdown  
# Download the dataset from Google Drive
gdown --id 14PptNxqI7D6YuTiPOjt1H0uLaa8Qr0qF  
# Unzip the dataset into the 'data' directory
unzip -q product_description_data.zip -d data  
```




## **🚀 Training the Model**  
To fine-tune the model, run:  
```bash
python main.py
```
This script:  
✅ Loads **LLaMA 3.2-11B-Vision**  
✅ Applies **LoRA** for efficient fine-tuning  
✅ Saves the fine-tuned model



## **🔍 Running Inference on a Single Image**  
Use the pretrained model to generate a product description:  
```bash
python inference.py "path/to/image.jpg"
```

## **Acknowledgement**
This project is based on the [Unsloth](https://github.com/unslothai/unsloth).