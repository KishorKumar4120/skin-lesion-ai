# streamlit_app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# ============================================================
# MODEL ARCHITECTURE (Must match the trained weights)
# ============================================================
from torchvision import models
import torch.nn as nn

def load_model():
    """Load the ResNet18 model with trained weights"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)
    
    # Load weights from the same directory
    weights_path = "model.pth"
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        st.error("Model file not found! Please upload model.pth")
        return None
    
    model.eval()
    return model

# Class labels
CLASS_LABELS = {
    0: "Actinic keratoses - PRECANCEROUS",
    1: "Basal cell carcinoma - CANCEROUS",
    2: "Benign keratosis - BENIGN",
    3: "Dermatofibroma - BENIGN",
    4: "Melanocytic nevi - BENIGN (common mole)",
    5: "Vascular lesions - BENIGN",
    6: "Melanoma - DANGEROUS (skin cancer)"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# GRAD-CAM Implementation
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Skin Lesion AI", page_icon="🩺", layout="wide")

st.title("🩺 AI-Powered Skin Lesion Diagnostic Assistant")
st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.header("📋 About")
    st.info("""
    This AI model is trained on the **HAM10000 dataset** and can classify 
    7 types of skin lesions using a **ResNet18** deep learning architecture.
    
    **⚠️ Important:**
    - For research/educational purposes only
    - NOT a medical device
    - Always consult a dermatologist
    """)
    
    st.header("📊 Skin Lesion Types")
    for idx, label in CLASS_LABELS.items():
        if "BENIGN" in label:
            st.success(f"🟢 {label}")
        elif "PRECANCEROUS" in label:
            st.warning(f"🟡 {label}")
        else:
            st.error(f"🔴 {label}")

# Load model (cached for performance)
@st.cache_resource
def get_model():
    with st.spinner("Loading AI model... This may take a moment."):
        return load_model()

model = get_model()

if model:
    # Main area - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a skin lesion for AI analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("🔬 Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess
                img_tensor = transform(image).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                class_idx = predicted.item()
                diagnosis = CLASS_LABELS[class_idx]
                confidence_score = confidence.item()
                
                # Display diagnosis
                if "BENIGN" in diagnosis:
                    st.success(f"### 🎯 {diagnosis}")
                elif "PRECANCEROUS" in diagnosis:
                    st.warning(f"### 🎯 {diagnosis}")
                else:
                    st.error(f"### 🎯 {diagnosis}")
                
                st.metric("Confidence Score", f"{confidence_score:.2%}")
                
                # Show top probabilities
                st.markdown("### 📊 Probability Distribution")
                probs_with_labels = list(zip(probabilities[0].tolist(), CLASS_LABELS.values()))
                probs_with_labels.sort(reverse=True)
                
                for prob, label in probs_with_labels[:5]:
                    if "BENIGN" in label:
                        color = "green"
                    elif "PRECANCEROUS" in label:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(f"**{label.split('-')[0].strip()}**")
                    st.progress(prob, text=f"{prob:.2%}")
                
                # Grad-CAM button
                if st.button("🔍 Show AI Attention Map (Grad-CAM)"):
                    with st.spinner("Generating heatmap..."):
                        # Find target layer
                        if hasattr(model, 'layer4'):
                            target_layer = model.layer4[-1]
                        else:
                            for module in model.modules():
                                if isinstance(module, torch.nn.Conv2d):
                                    target_layer = module
                                    break
                        
                        grad_cam = GradCAM(model, target_layer)
                        heatmap, _ = grad_cam.generate_heatmap(img_tensor, class_idx)
                        
                        # Prepare original image
                        orig_np = np.array(image.resize((224, 224))) / 255.0
                        
                        # Resize and colorize heatmap
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                        
                        # Overlay
                        orig_uint8 = (orig_np * 255).astype(np.uint8)
                        overlayed = cv2.addWeighted(orig_uint8, 0.6, heatmap_colored, 0.4, 0)
                        
                        # Display
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(orig_uint8)
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title("Attention Heatmap")
                        axes[1].axis('off')
                        
                        axes[2].imshow(overlayed)
                        axes[2].set_title(f"Focus: {diagnosis.split('-')[0][:20]}")
                        axes[2].axis('off')
                        
                        st.pyplot(fig)
                        plt.close()
            
        else:
            st.info("👈 Upload an image to start analysis")

# Footer
st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This tool is for educational purposes only. Not for clinical diagnosis. Always consult a qualified healthcare provider.")