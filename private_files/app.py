import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #c62828;
    }
    .normal-alert {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_data
def load_models():
    """Load all saved models and preprocessing objects"""
    try:
        # Load the best model
        best_model = joblib.load('best_fraud_model.pkl')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load feature names
        feature_names = joblib.load('feature_names.pkl')
        
        # Load model summary
        summary = joblib.load('model_summary.pkl')
        
        return best_model, scaler, feature_names, summary
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the Jupyter notebook first to train and save the models.")
        return None, None, None, None

def generate_sample_transaction():
    """Generate a random sample transaction for demo purposes"""
    np.random.seed(42)
    
    # Generate realistic transaction data
    sample_data = {
        'Time': np.random.uniform(0, 172792),
        'V1': np.random.normal(0, 1),
        'V2': np.random.normal(0, 1),
        'V3': np.random.normal(0, 1),
        'V4': np.random.normal(0, 1),
        'V5': np.random.normal(0, 1),
        'V6': np.random.normal(0, 1),
        'V7': np.random.normal(0, 1),
        'V8': np.random.normal(0, 1),
        'V9': np.random.normal(0, 1),
        'V10': np.random.normal(0, 1),
        'V11': np.random.normal(0, 1),
        'V12': np.random.normal(0, 1),
        'V13': np.random.normal(0, 1),
        'V14': np.random.normal(0, 1),
        'V15': np.random.normal(0, 1),
        'V16': np.random.normal(0, 1),
        'V17': np.random.normal(0, 1),
        'V18': np.random.normal(0, 1),
        'V19': np.random.normal(0, 1),
        'V20': np.random.normal(0, 1),
        'V21': np.random.normal(0, 1),
        'V22': np.random.normal(0, 1),
        'V23': np.random.normal(0, 1),
        'V24': np.random.normal(0, 1),
        'V25': np.random.normal(0, 1),
        'V26': np.random.normal(0, 1),
        'V27': np.random.normal(0, 1),
        'V28': np.random.normal(0, 1),
        'Amount': np.random.lognormal(3, 1.5)
    }
    
    return sample_data

def predict_fraud(transaction_data, model, scaler, feature_names):
    """Predict fraud for a single transaction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Ensure all features are present and in correct order
        df = df[feature_names]
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_data)[0]
            fraud_probability = proba[1]  # Probability of fraud
        else:
            # For models without predict_proba (like Isolation Forest)
            fraud_probability = 0.5  # Default value
        
        return prediction, fraud_probability
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load models
    best_model, scaler, feature_names, summary = load_models()
    
    if best_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Display model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.metric("Best Model", summary['best_model'])
    st.sidebar.metric("F1-Score", f"{summary['best_f1_score']:.4f}")
    st.sidebar.metric("Precision", f"{summary['best_precision']:.4f}")
    st.sidebar.metric("Recall", f"{summary['best_recall']:.4f}")
    st.sidebar.metric("AUC", f"{summary['best_auc']:.4f}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Single Transaction Fraud Detection")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Sample Transaction", "Upload CSV"]
        )
        
        if input_method == "Manual Input":
            st.subheader("Enter Transaction Details")
            
            # Create input fields
            col1, col2 = st.columns(2)
            
            with col1:
                time = st.number_input("Time (seconds)", value=0.0, step=1.0)
                amount = st.number_input("Amount ($)", value=0.0, step=0.01, format="%.2f")
                
                # V1-V14 features
                v1 = st.number_input("V1", value=0.0, step=0.01, format="%.4f")
                v2 = st.number_input("V2", value=0.0, step=0.01, format="%.4f")
                v3 = st.number_input("V3", value=0.0, step=0.01, format="%.4f")
                v4 = st.number_input("V4", value=0.0, step=0.01, format="%.4f")
                v5 = st.number_input("V5", value=0.0, step=0.01, format="%.4f")
                v6 = st.number_input("V6", value=0.0, step=0.01, format="%.4f")
                v7 = st.number_input("V7", value=0.0, step=0.01, format="%.4f")
                v8 = st.number_input("V8", value=0.0, step=0.01, format="%.4f")
                v9 = st.number_input("V9", value=0.0, step=0.01, format="%.4f")
                v10 = st.number_input("V10", value=0.0, step=0.01, format="%.4f")
                v11 = st.number_input("V11", value=0.0, step=0.01, format="%.4f")
                v12 = st.number_input("V12", value=0.0, step=0.01, format="%.4f")
                v13 = st.number_input("V13", value=0.0, step=0.01, format="%.4f")
                v14 = st.number_input("V14", value=0.0, step=0.01, format="%.4f")
            
            with col2:
                v15 = st.number_input("V15", value=0.0, step=0.01, format="%.4f")
                v16 = st.number_input("V16", value=0.0, step=0.01, format="%.4f")
                v17 = st.number_input("V17", value=0.0, step=0.01, format="%.4f")
                v18 = st.number_input("V18", value=0.0, step=0.01, format="%.4f")
                v19 = st.number_input("V19", value=0.0, step=0.01, format="%.4f")
                v20 = st.number_input("V20", value=0.0, step=0.01, format="%.4f")
                v21 = st.number_input("V21", value=0.0, step=0.01, format="%.4f")
                v22 = st.number_input("V22", value=0.0, step=0.01, format="%.4f")
                v23 = st.number_input("V23", value=0.0, step=0.01, format="%.4f")
                v24 = st.number_input("V24", value=0.0, step=0.01, format="%.4f")
                v25 = st.number_input("V25", value=0.0, step=0.01, format="%.4f")
                v26 = st.number_input("V26", value=0.0, step=0.01, format="%.4f")
                v27 = st.number_input("V27", value=0.0, step=0.01, format="%.4f")
                v28 = st.number_input("V28", value=0.0, step=0.01, format="%.4f")
            
            # Create transaction data
            transaction_data = {
                'Time': time, 'Amount': amount,
                'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5, 'V6': v6, 'V7': v7, 'V8': v8,
                'V9': v9, 'V10': v10, 'V11': v11, 'V12': v12, 'V13': v13, 'V14': v14,
                'V15': v15, 'V16': v16, 'V17': v17, 'V18': v18, 'V19': v19, 'V20': v20,
                'V21': v21, 'V22': v22, 'V23': v23, 'V24': v24, 'V25': v25, 'V26': v26,
                'V27': v27, 'V28': v28
            }
        
        elif input_method == "Sample Transaction":
            st.subheader("Sample Transaction")
            if st.button("Generate Random Transaction"):
                transaction_data = generate_sample_transaction()
                
                # Display the generated transaction
                st.write("Generated Transaction Data:")
                df_display = pd.DataFrame([transaction_data])
                st.dataframe(df_display, use_container_width=True)
        
        elif input_method == "Upload CSV":
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check if required columns are present
                    required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                    missing_columns = [col for col in required_columns if col not in df_upload.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {missing_columns}")
                    else:
                        st.write("Uploaded Data Preview:")
                        st.dataframe(df_upload.head(), use_container_width=True)
                        
                        # Use first row for prediction
                        if len(df_upload) > 0:
                            transaction_data = df_upload.iloc[0].to_dict()
                            st.success("Using first row for prediction")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    transaction_data = None
        
        # Prediction button and results
        if st.button("🔍 Predict Fraud", type="primary") and 'transaction_data' in locals():
            with st.spinner("Analyzing transaction..."):
                prediction, fraud_probability = predict_fraud(transaction_data, best_model, scaler, feature_names)
            
            if prediction is not None:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="fraud-alert"><h3>🚨 FRAUD DETECTED</h3><p>This transaction has been flagged as fraudulent.</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="normal-alert"><h3>✅ NORMAL TRANSACTION</h3><p>This transaction appears to be legitimate.</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Fraud Probability", f"{fraud_probability:.2%}")
                
                with col3:
                    st.metric("Prediction", "Fraud" if prediction == 1 else "Normal")
                
                # Show transaction details
                st.subheader("Transaction Details")
                df_result = pd.DataFrame([transaction_data])
                st.dataframe(df_result, use_container_width=True)
                
                # Risk assessment
                st.subheader("Risk Assessment")
                if fraud_probability >= 0.7:
                    risk_level = "High Risk"
                    risk_color = "red"
                elif fraud_probability >= 0.4:
                    risk_level = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk"
                    risk_color = "green"
                
                st.markdown(f"**Risk Level:** <span style='color: {risk_color}; font-weight: bold;'>{risk_level}</span>", unsafe_allow_html=True)
    
    with tab2:
        st.header("Batch Analysis")
        st.write("Upload a CSV file to analyze multiple transactions at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file for batch analysis", type="csv", key="batch_upload")
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                missing_columns = [col for col in required_columns if col not in df_batch.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    st.write(f"Analyzing {len(df_batch)} transactions...")
                    
                    # Process each transaction
                    predictions = []
                    probabilities = []
                    
                    with st.spinner("Processing transactions..."):
                        for idx, row in df_batch.iterrows():
                            pred, prob = predict_fraud(row.to_dict(), best_model, scaler, feature_names)
                            predictions.append(pred)
                            probabilities.append(prob)
                    
                    # Add predictions to dataframe
                    df_batch['Prediction'] = ['Fraud' if p == 1 else 'Normal' for p in predictions]
                    df_batch['Fraud_Probability'] = probabilities
                    
                    # Display results
                    st.subheader("Batch Analysis Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(df_batch))
                    with col2:
                        fraud_count = sum(predictions)
                        st.metric("Fraudulent", fraud_count)
                    with col3:
                        st.metric("Normal", len(df_batch) - fraud_count)
                    with col4:
                        fraud_rate = fraud_count / len(df_batch)
                        st.metric("Fraud Rate", f"{fraud_rate:.2%}")
                    
                    # Display results table
                    st.write("Detailed Results:")
                    st.dataframe(df_batch, use_container_width=True)
                    
                    # Download results
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("Visualizations")
                    
                    # Fraud distribution
                    fig1 = px.pie(
                        values=[fraud_count, len(df_batch) - fraud_count],
                        names=['Fraud', 'Normal'],
                        title="Fraud Distribution"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Fraud probability distribution
                    fig2 = px.histogram(
                        df_batch, x='Fraud_Probability',
                        title="Distribution of Fraud Probabilities",
                        nbins=20
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing batch file: {e}")
    
    with tab3:
        st.header("About the Fraud Detection System")
        
        st.markdown("""
        ### 🎯 Project Overview
        This fraud detection system uses machine learning to identify fraudulent credit card transactions. 
        The system was trained on a dataset containing anonymized credit card transactions and uses multiple 
        algorithms to achieve high accuracy in fraud detection.
        
        ### 🤖 Models Used
        - **Logistic Regression**: Linear classifier for baseline performance
        - **Random Forest**: Ensemble method with feature importance analysis
        - **XGBoost**: Gradient boosting for high performance
        - **Isolation Forest**: Unsupervised anomaly detection
        
        ### 📊 Model Performance
        """)
        
        # Display model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Model", summary['best_model'])
            st.metric("F1-Score", f"{summary['best_f1_score']:.4f}")
            st.metric("Precision", f"{summary['best_precision']:.4f}")
        
        with col2:
            st.metric("Recall", f"{summary['best_recall']:.4f}")
            st.metric("AUC", f"{summary['best_auc']:.4f}")
            st.metric("Dataset Size", f"{summary['dataset_size']:,}")
        
        st.markdown("""
        ### 🔧 Technical Details
        - **Data Preprocessing**: RobustScaler for feature scaling
        - **Class Imbalance**: SMOTE for oversampling minority class
        - **Feature Engineering**: 30 features including Time, Amount, and 28 PCA components
        - **Evaluation**: Cross-validation with stratified sampling
        
        ### 📈 Key Features
        - Real-time fraud detection
        - Batch processing capabilities
        - Probability scoring for risk assessment
        - Comprehensive model evaluation metrics
        - Interactive web interface
        
        ### ⚠️ Important Notes
        - This is a demonstration system for educational purposes
        - Real fraud detection systems require additional security measures
        - Model performance may vary with different datasets
        - Regular retraining is recommended for production use
        """)

if __name__ == "__main__":
    main()
