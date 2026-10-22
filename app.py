import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Page configuration
st.set_page_config(
    page_title="Wheat Disease Classification",
    page_icon="🌾",
    layout="wide"
)

# Disease information dictionary
DISEASE_INFO = {
    "Crown and Root Rot": {
        "description": "A disease complex caused by Fusarium species and other soilborne pathogens. Affects the crown (base of the stem) and root system. Symptoms include brown or black lesions at the base of the stem, yellowing of leaves, stunted growth, and sometimes 'whiteheads' (heads that die prematurely without filling grain).",
        "treatment": "1. Apply seed treatment with fungicides before planting\n2. Use crop rotation with non-host crops\n3. Improve soil drainage\n4. Avoid excessive irrigation\n5. Use resistant wheat varieties when available",
        "precautions": "Prevention is key as the disease is difficult to control once established. Focus on seed treatment and crop rotation."
    },
    "Healthy Wheat": {
        "description": "No visible signs of disease or stress. Normal green foliage, well-formed tillers, and typical growth characteristics. Note that visual inspection alone cannot guarantee a completely disease-free plant; subsurface infections may not yet show symptoms.",
        "treatment": "No treatment needed. Continue regular monitoring and maintenance practices.",
        "precautions": "Maintain good agricultural practices to prevent future disease outbreaks."
    },
    "Leaf Rust (Brown Rust)": {
        "description": "Caused by the fungus Puccinia triticina. Appears as small, circular to elliptical orange-brown powdery pustules (uredia) mainly on the upper surfaces of leaves and leaf sheaths. Heavy infection reduces photosynthetic area, leading to lower grain weight and yield.",
        "treatment": "1. Apply foliar fungicides at early infection stages\n2. Use resistant wheat varieties\n3. Remove volunteer wheat plants\n4. Apply fungicides preventatively in high-risk areas",
        "precautions": "Disease spreads rapidly by windborne spores in warm, moist conditions. Monitor fields regularly during humid periods."
    },
    "Septoria": {
        "description": "A group of fungal diseases. Septoria tritici blotch is caused by Zymoseptoria tritici; Septoria nodorum blotch is caused by Parastagonospora nodorum. Symptoms appear as tan or brown irregular leaf blotches, often containing tiny black dots called pycnidia (spore-producing structures).",
        "treatment": "1. Apply fungicides containing triazoles or strobilurins\n2. Use resistant varieties\n3. Practice crop rotation\n4. Remove infected crop residues\n5. Avoid late planting",
        "precautions": "Prolonged wet, humid weather favors infection. Can cause serious yield losses if the flag leaf is heavily affected."
    },
    "Stripe Rust (Yellow Rust)": {
        "description": "Caused by Puccinia striiformis f. sp. tritici. Characterized by narrow yellow to orange-yellow stripes of powdery pustules running parallel to the leaf veins. Pustules may also appear on leaf sheaths, glumes, and necks.",
        "treatment": "1. Apply fungicides at first sign of infection\n2. Use resistant wheat varieties\n3. Early planting can help avoid peak infection periods\n4. Apply protective fungicides in high-risk regions",
        "precautions": "Thrives in cool, moist conditions. Severe infections can significantly reduce grain yield. One of the most economically damaging wheat diseases globally."
    }
}

# Model configurations
MODEL_CONFIGS = {
    "ResNet50": {
        "path": "models/resnet50_balanced_500.h5",
        "preprocess": resnet_preprocess,
        "accuracy": 91.6
    },
    "EfficientNetV2B0": {
        "path": "models/efficientnetv2b0_balanced_500.h5",
        "preprocess": efficientnetv2_preprocess,
        "accuracy": 91.2
    },
    "MobileNetV2": {
        "path": "models/mobilenetv2_balanced_500.h5",
        "preprocess": mobilenet_preprocess,
        "accuracy": 88.4
    },
    "EfficientNetB0": {
        "path": "models/efficientnetb0_balanced_500.h5",
        "preprocess": efficientnet_preprocess,
        "accuracy": 80.4
    }
}

# Class labels
CLASS_LABELS = ["Crown and Root Rot", "Healthy Wheat", "Leaf Rust (Brown Rust)", "Septoria", "Stripe Rust (Yellow Rust)"]

# Load all models
@st.cache_resource
def load_all_models():
    models = {}
    for model_name, config in MODEL_CONFIGS.items():
        try:
            model = tf.keras.models.load_model(
                config["path"],
                custom_objects={'preprocess_input': config["preprocess"]}
            )
            models[model_name] = model
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
    return models

# Preprocess image
def preprocess_image(image, preprocess_func):
    # Correct camera orientation
    img = ImageOps.exif_transpose(image)
    
    # Ensure RGB
    img = img.convert("RGB")
    
    # Center square crop
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    
    # Resize to 224x224
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Apply model-specific preprocessing
    img_array = preprocess_func(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# TTA augmentation - create 7 augmented versions
def apply_tta(image, preprocess_func):
    # Correct camera orientation
    img = ImageOps.exif_transpose(image)
    img = img.convert("RGB")
    
    # Center square crop
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    augmentations = []
    
    # 1. Original
    augmentations.append(img)
    
    # 2. Horizontal flip
    augmentations.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # 3. Brightness +0.05
    aug = img.copy()
    aug = tf.image.adjust_brightness(np.array(aug), 0.05)
    augmentations.append(Image.fromarray(np.clip(aug.numpy(), 0, 255).astype(np.uint8)))
    
    # 4. Brightness -0.05
    aug = img.copy()
    aug = tf.image.adjust_brightness(np.array(aug), -0.05)
    augmentations.append(Image.fromarray(np.clip(aug.numpy(), 0, 255).astype(np.uint8)))
    
    # 5. Contrast ×1.1
    aug = img.copy()
    aug = tf.image.adjust_contrast(np.array(aug), 1.1)
    augmentations.append(Image.fromarray(np.clip(aug.numpy(), 0, 255).astype(np.uint8)))
    
    # 6. Contrast ×0.9
    aug = img.copy()
    aug = tf.image.adjust_contrast(np.array(aug), 0.9)
    augmentations.append(Image.fromarray(np.clip(aug.numpy(), 0, 255).astype(np.uint8)))
    
    # 7. Central crop (90%)
    aug = img.copy()
    w, h = aug.size
    crop_size = int(min(w, h) * 0.9)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    aug = aug.crop((left, top, left + crop_size, top + crop_size))
    aug = aug.resize((224, 224), Image.Resampling.LANCZOS)
    augmentations.append(aug)
    
    # Preprocess all augmentations
    processed = []
    for aug_img in augmentations:
        img_array = np.array(aug_img)
        img_array = preprocess_func(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        processed.append(img_array)
    
    return np.concatenate(processed, axis=0)

# Main app
def main():
    st.title("🌾 Wheat Disease Classification")
    st.markdown("---")
    
    # Load all models
    models = load_all_models()
    
    if not models:
        st.error("No models could be loaded. Please ensure model files exist in the 'models/' directory.")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Ensemble (All Models)"] + list(models.keys()),
        index=0,
        help="Choose which model to use for prediction. Ensemble combines all models for better accuracy."
    )
    
    # Use ensemble or single model
    use_ensemble = selected_model == "Ensemble (All Models)"
    model = models[list(models.keys())[0]] if use_ensemble else models[selected_model]
    
    # Sidebar
    st.sidebar.title("About")
    if use_ensemble:
        avg_accuracy = sum(config["accuracy"] for config in MODEL_CONFIGS.values()) / len(MODEL_CONFIGS)
        st.sidebar.info(f"""
        This app uses deep learning models trained on 2,500 wheat images to classify wheat diseases.
        
        **Current Mode: Ensemble (All 4 Models)**
        **Average Training Accuracy: {avg_accuracy:.1f}%**
        
        Ensemble combines predictions from all models for more accurate and reliable results.
        """)
    else:
        st.sidebar.info(f"""
        This app uses deep learning models trained on 2,500 wheat images to classify wheat diseases.
        
        **Current Model: {selected_model}**
        **Training Accuracy: {MODEL_CONFIGS[selected_model]['accuracy']}%**
        
        The model can identify 5 different classes of wheat conditions.
        """)
    
    # File upload
    st.header("Upload Wheat Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of wheat leaves or plant for disease classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        # Preprocess and predict with TTA option
        use_tta = st.checkbox("Use Test Time Augmentation (TTA)", value=False, 
                            help="TTA averages predictions from 7 augmented versions of the image for more robust results")
        
        # Preprocess and predict
        with st.spinner("Analyzing image..."):
            if use_ensemble:
                # Get predictions from all models
                all_predictions = []
                for model_name, current_model in models.items():
                    preprocess_func = MODEL_CONFIGS[model_name]["preprocess"]
                    
                    if use_tta:
                        tta_images = apply_tta(image, preprocess_func)
                        tta_preds = current_model.predict(tta_images, verbose=0)
                        preds = np.mean(tta_preds, axis=0, keepdims=True)
                    else:
                        processed_image = preprocess_image(image, preprocess_func)
                        preds = current_model.predict(processed_image, verbose=0)
                    
                    all_predictions.append(preds[0])
                
                # Average predictions from all models (weighted average by training accuracy)
                weights = np.array([MODEL_CONFIGS[name]["accuracy"] for name in models.keys()])
                weights = weights / weights.sum()  # Normalize weights
                predictions = np.average(all_predictions, axis=0, weights=weights)
                predictions = np.expand_dims(predictions, axis=0)
                
                st.info(f"🎯 Ensemble: Combined predictions from {len(models)} models (weighted by accuracy)")
            else:
                # Single model prediction
                preprocess_func = MODEL_CONFIGS[selected_model]["preprocess"]
                
                if use_tta:
                    tta_images = apply_tta(image, preprocess_func)
                    tta_predictions = model.predict(tta_images, verbose=0)
                    predictions = np.mean(tta_predictions, axis=0, keepdims=True)
                    st.info("🔍 TTA applied: Averaging predictions from 7 augmented versions")
                else:
                    processed_image = preprocess_image(image, preprocess_func)
                    predictions = model.predict(processed_image, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = CLASS_LABELS[predicted_class_idx]
        
        # Display prediction
        with col2:
            st.subheader("Prediction Result")
            
            # Prediction card
            confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #f0f0f0; border: 2px solid {confidence_color};">
                <h2 style="margin: 0; color: {confidence_color};">{predicted_class}</h2>
                <p style="margin: 10px 0 0 0; font-size: 24px; color: #333333;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence warning
            if confidence < 0.6:
                st.warning(f"⚠️ Low confidence ({confidence:.1%}). The model is uncertain. Check the confidence breakdown below and consider predictions from other models in the comparison table.")
            elif confidence < 0.8:
                st.info(f"ℹ️ Moderate confidence ({confidence:.1%}). The prediction is reasonably confident but consider checking other models' predictions.")
        
        # Disease information
        st.markdown("---")
        st.header("Disease Information")
        
        info = DISEASE_INFO[predicted_class]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Description")
            st.write(info["description"])
            
            st.subheader("Treatment")
            st.info(info["treatment"])
        
        with col2:
            st.subheader("Precautions")
            st.warning(info["precautions"])
        
        # Confidence breakdown
        st.markdown("---")
        st.header("Confidence Breakdown")
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(predictions[0]),
                y=CLASS_LABELS,
                orientation='h',
                marker_color=['#2ecc71' if i == predicted_class_idx else '#95a5a6' for i in range(len(CLASS_LABELS))]
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence for All Classes",
            xaxis_title="Confidence",
            yaxis_title="Disease Class",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement analysis
        st.markdown("---")
        st.header("Model Agreement Analysis")
        
        # Get predictions from all models for agreement analysis
        model_predictions = {}
        for model_name, current_model in models.items():
            preprocess_func = MODEL_CONFIGS[model_name]["preprocess"]
            processed_image = preprocess_image(image, preprocess_func)
            pred = current_model.predict(processed_image, verbose=0)
            pred_class_idx = np.argmax(pred[0])
            pred_class = CLASS_LABELS[pred_class_idx]
            pred_confidence = float(pred[0][pred_class_idx])
            model_predictions[model_name] = {
                "class": pred_class,
                "confidence": pred_confidence
            }
        
        # Count votes for each class
        votes = {}
        for model_name, pred in model_predictions.items():
            if pred["class"] not in votes:
                votes[pred["class"]] = 0
            votes[pred["class"]] += 1
        
        # Find most common prediction
        most_common_class = max(votes.items(), key=lambda x: x[1])
        
        # Check if models agree
        if len(votes) == 1:
            st.success(f"✅ All models agree on: **{most_common_class[0]}**")
            st.write(f"All {len(models)} models ({', '.join(models.keys())}) predict: {most_common_class[0]}")
        else:
            st.warning(f"⚠️ Models disagree! Most common prediction: **{most_common_class[0]}** ({most_common_class[1]}/{len(models)} models)")
            
            # Show which models agree
            agreeing_models = [name for name, pred in model_predictions.items() if pred["class"] == most_common_class[0]]
            st.success(f"**Models agreeing with {most_common_class[0]}:**")
            for model_name in agreeing_models:
                conf = model_predictions[model_name]["confidence"]
                st.write(f"- {model_name}: {conf:.1%}")
            
            # Show which models disagree
            disagreeing_models = [(name, pred) for name, pred in model_predictions.items() if pred["class"] != most_common_class[0]]
            if disagreeing_models:
                st.error(f"**Models disagreeing:**")
                for model_name, pred in disagreeing_models:
                    st.write(f"- {model_name}: predicts '{pred['class']}' ({pred['confidence']:.1%})")
        
        # Find most confident model
        most_confident_model = max(model_predictions.items(), key=lambda x: x[1]["confidence"])
        
        # Suggest most confident model if it differs from current selection
        if most_confident_model[1]["confidence"] > confidence + 0.1:  # If >10% more confident
            st.info(f"💡 **Recommendation**: {most_confident_model[0]} is most confident ({most_confident_model[1]['confidence']:.1%}) with prediction '{most_confident_model[1]['class']}'. Consider switching to this model in the sidebar.")
    
    # Model comparison (expandable)
    with st.expander("View Model Comparison"):
        st.subheader("Real-time Predictions from All Models")
        
        if uploaded_file is not None:
            st.info("📊 Showing predictions from all 4 models on the uploaded image")
            
            comparison_data = []
            for model_name, model in models.items():
                # Get preprocessing function
                preprocess_func = MODEL_CONFIGS[model_name]["preprocess"]
                
                # Predict with this model
                processed_image = preprocess_image(image, preprocess_func)
                pred = model.predict(processed_image, verbose=0)
                pred_class_idx = np.argmax(pred[0])
                pred_class = CLASS_LABELS[pred_class_idx]
                pred_confidence = float(pred[0][pred_class_idx])
                
                row = {
                    "Model": model_name,
                    "Prediction": pred_class,
                    "Confidence": f"{pred_confidence:.1%}",
                    "Training Accuracy": f"{MODEL_CONFIGS[model_name]['accuracy']}%"
                }
                comparison_data.append(row)
            
            st.table(comparison_data)
        else:
            st.warning("Upload an image to see real-time predictions from all models")
            
            # Show static training accuracy when no image is uploaded
            st.subheader("Training Accuracy (Test Set)")
            training_data = []
            for model_name, config in MODEL_CONFIGS.items():
                training_data.append({
                    "Model": model_name,
                    "Test Accuracy": f"{config['accuracy']}%"
                })
            st.table(training_data)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Disclaimer**
    
    Disease identification from an image is preliminary. Symptoms of different wheat diseases can overlap visually. 
    Always confirm uncertain diagnoses with a qualified agricultural expert or plant pathologist. 
    Follow current local pesticide registration and product-label instructions before applying any treatment.
    """)

if __name__ == "__main__":
    main()
