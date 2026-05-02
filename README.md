\# 🩺 Skin Lesion AI Classifier



\[!\[Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/KishorKumar4120/skin-lesion-ai)

\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat\&logo=Streamlit\&logoColor=white)](https://streamlit.io)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=PyTorch\&logoColor=white)](https://pytorch.org)



\## 📋 Overview



An AI-powered web application that classifies skin lesions into 7 categories using a ResNet18 deep learning model fine-tuned on the HAM10000 dataset.



\*\*Live Demo:\*\* \[https://huggingface.co/spaces/KishorKumar4120/skin-lesion-ai](https://huggingface.co/spaces/KishorKumar4120/skin-lesion-ai)



\## 🧬 Skin Lesion Types



| Class | Type | Risk Level |

|-------|------|------------|

| akk | Actinic keratoses | 🟡 PRECANCEROUS |

| bcc | Basal cell carcinoma | 🔴 CANCEROUS |

| bkl | Benign keratosis | 🟢 BENIGN |

| df | Dermatofibroma | 🟢 BENIGN |

| mel | Melanoma | 🔴 DANGEROUS |

| nv | Melanocytic nevi | 🟢 BENIGN |

| vasc | Vascular lesions | 🟢 BENIGN |



\## 🚀 Features



\- ✅ Real-time skin lesion classification

\- ✅ 7-class prediction with confidence scores

\- ✅ Probability distribution visualization

\- ✅ User-friendly web interface



\## 🛠️ Tech Stack



| Technology | Purpose |

|------------|---------|

| Python 3.10+ | Backend language |

| PyTorch | Deep learning framework |

| ResNet18 | Model architecture |

| Streamlit | Web application framework |

| Hugging Face Spaces | Deployment platform |



\## 📊 Model Performance



\- \*\*Dataset:\*\* HAM10000 (10,015 dermoscopic images)

\- \*\*Architecture:\*\* ResNet18

\- \*\*Input size:\*\* 224×224 pixels

\- \*\*Classes:\*\* 7 skin lesion types



\## 🚀 Local Development



\### Installation



```bash

\# Clone the repository

git clone https://github.com/YOUR\_USERNAME/skin-lesion-ai.git

cd skin-lesion-ai



\# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements.txt



\# Run the app

streamlit run streamlit\_app.py


📁 Project Structure
skin-lesion-ai/

├── streamlit_app.py      # Main application
├── requirements.txt      # Python dependencies
├── model.pth            # Pre-trained weights
└── README.md            # Project documentation

⚠️ Medical Disclaimer
This tool is for educational and research purposes only. It is NOT a medical device and not FDA-approved. Always consult a qualified dermatologist for proper diagnosis.

📈 Future Improvements
Add Grad-CAM visualization

Support batch processing

PDF report generation

📄 License
MIT License - feel free to use and modify!

🙏 Acknowledgments
HAM10000 dataset creators

Kalbe Digital Lab for model inspiration

Hugging Face for free hosting

📧 Contact
Kishor Kumar - Your GitHub Profile

Project Link: https://github.com/KishorKumar4120/skin-lesion-ai

