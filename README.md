**SKIN LESION AI CLASSIFIER**
===============================================================================

**WEB APP BADGES**

🤗 Live Demo: https://huggingface.co/spaces/KishorKumar4120/skin-lesion-ai
GitHub Repo: https://github.com/KishorKumar4120/skin-lesion-ai
Python: 3.10+
PyTorch: Latest
Streamlit: Latest

AI-Powered Dermatology Assistant | 7 Lesion Types | Research Grade

TRY LIVE DEMO: https://huggingface.co/spaces/KishorKumar4120/skin-lesion-ai


ABOUT
===============================================================================

This deep learning model analyzes skin lesion images and classifies them into 7 different skin conditions - from benign moles to malignant melanomas.

Built with: ResNet18 + PyTorch + Streamlit
Trained on: HAM10000 dataset (10,015 medical-grade images)
Deployed on: Hugging Face Spaces


SKIN LESION TYPES
===============================================================================

BENIGN (Green)	PRECANCEROUS (Yellow)	MALIGNANT (Red)
Benign Keratosis	Actinic Keratoses	Basal Cell Carcinoma
Dermatofibroma		Melanoma
Melanocytic Nevi (Moles)		
Vascular Lesions		

FEATURES
===============================================================================

✔ Real-time predictions - Results in under 2 seconds
✔ Confidence scores - See how certain the AI is
✔ Probability breakdown - Full distribution across all 7 classes
✔ Clean interface - Easy to use, mobile-friendly
✔ Free access - No signup required


TECH STACK
===============================================================================

Category	Technology
Framework	PyTorch
Architecture	ResNet18
Web App	Streamlit
Deployment	Hugging Face Spaces
Dataset	HAM10000

RUN LOCALLY
===============================================================================

Step 1: Clone the repository
git clone https://github.com/KishorKumar4120/skin-lesion-ai.git
cd skin-lesion-ai

Step 2: Create virtual environment
python -m venv venv

On Windows:
venv\Scripts\activate

On Mac/Linux:
source venv/bin/activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Run the application
streamlit run streamlit_app.py


MODEL PERFORMANCE
===============================================================================

Metric	Score
Accuracy	88.2%
Precision	87.5%
Recall	86.9%
F1-Score	87.2%

PROJECT STRUCTURE
===============================================================================

skin-lesion-ai/

              ├── streamlit_app.py # Main application
              
              ├── requirements.txt # Python dependencies
              
              ├── model.pth # Pre-trained weights
              
              ├── README.md # Documentation
              
              └── sample_images/ # Test images


MEDICAL DISCLAIMER
===============================================================================

FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY

NOT FDA-approved
NOT a medical device

Always consult a qualified dermatologist for any skin concerns.


CONTACT
===============================================================================

Kishor Kumar

GitHub: https://github.com/KishorKumar4120


===============================================================================

Star this repo if you found it useful!

Built with love for AI in Healthcare

===============================================================================


