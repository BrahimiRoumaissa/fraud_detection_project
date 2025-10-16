"""
Fraud Detection System - Demo Application

This is a simplified demo version of the Streamlit application
that showcases the interface and functionality without exposing
the full ML implementation details.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System - Demo",
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

def generate_sample_transaction():
    """Generate a random sample transaction for demo purposes"""
    np.random.seed(42)
    
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

def demo_prediction(transaction_data):
    """
    Demo prediction function - simulates ML model prediction
    In the actual implementation, this would use trained models
    """
    # Simulate model prediction based on amount and some features
    amount = transaction_data.get('Amount', 0)
    v11 = transaction_data.get('V11', 0)
    v4 = transaction_data.get('V4', 0)
    
    # Simple demo logic (not the actual ML model)
    fraud_probability = 0.1 + (amount / 10000) * 0.3 + abs(v11) * 0.2 + abs(v4) * 0.1
    fraud_probability = min(fraud_probability, 0.95)  # Cap at 95%
    
    prediction = 1 if fraud_probability > 0.5 else 0
    
    return prediction, fraud_probability

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System - Demo</h1>', unsafe_allow_html=True)
    
    # Demo notice
    st.info("🚨 **Demo Version**: This is a simplified demonstration of the fraud detection system interface. The actual implementation includes trained ML models and full functionality.")
    
    # Sidebar
    st.sidebar.title("🔧 System Information")
    
    # Display demo model info
    st.sidebar.markdown("### Demo Model Performance")
    st.sidebar.metric("Best Model", "XGBoost")
    st.sidebar.metric("F1-Score", "0.8543")
    st.sidebar.metric("Precision", "0.8234")
    st.sidebar.metric("Recall", "0.8876")
    st.sidebar.metric("AUC", "0.9234")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Single Transaction Fraud Detection")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Sample Transaction"]
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
        
        # Prediction button and results
        if st.button("🔍 Predict Fraud", type="primary") and 'transaction_data' in locals():
            with st.spinner("Analyzing transaction..."):
                prediction, fraud_probability = demo_prediction(transaction_data)
            
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
        st.header("Batch Analysis Demo")
        st.write("This feature allows analyzing multiple transactions at once.")
        
        # Demo batch results
        st.subheader("Sample Batch Analysis Results")
        
        # Generate sample batch data
        np.random.seed(42)
        sample_batch = []
        for i in range(10):
            transaction = generate_sample_transaction()
            pred, prob = demo_prediction(transaction)
            transaction['Prediction'] = 'Fraud' if pred == 1 else 'Normal'
            transaction['Fraud_Probability'] = prob
            sample_batch.append(transaction)
        
        batch_df = pd.DataFrame(sample_batch)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(batch_df))
        with col2:
            fraud_count = (batch_df['Prediction'] == 'Fraud').sum()
            st.metric("Fraudulent", fraud_count)
        with col3:
            st.metric("Normal", len(batch_df) - fraud_count)
        with col4:
            fraud_rate = fraud_count / len(batch_df)
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
        
        # Display results table
        st.write("Sample Results:")
        st.dataframe(batch_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Sample Visualizations")
        
        # Fraud distribution
        fig1 = px.pie(
            values=[fraud_count, len(batch_df) - fraud_count],
            names=['Fraud', 'Normal'],
            title="Fraud Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Fraud probability distribution
        fig2 = px.histogram(
            batch_df, x='Fraud_Probability',
            title="Distribution of Fraud Probabilities",
            nbins=10
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("About the Fraud Detection System")
        
        st.markdown("""
        ### 🎯 Project Overview
        This fraud detection system uses machine learning to identify fraudulent credit card transactions. 
        The system was built using multiple algorithms and handles real-world challenges in fraud detection.
        
        ### 🤖 Technical Implementation
        - **Data Preprocessing**: RobustScaler for feature scaling, SMOTE for class imbalance
        - **ML Algorithms**: Logistic Regression, Random Forest, XGBoost, Isolation Forest
        - **Evaluation**: Comprehensive metrics including Precision, Recall, F1-Score, ROC-AUC
        - **Web Interface**: Interactive Streamlit application for real-time and batch processing
        
        ### 📊 Model Performance (Demo Values)
        """)
        
        # Display model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Model", "XGBoost")
            st.metric("F1-Score", "0.8543")
            st.metric("Precision", "0.8234")
        
        with col2:
            st.metric("Recall", "0.8876")
            st.metric("AUC", "0.9234")
            st.metric("Dataset Size", "284,807")
        
        st.markdown("""
        ### 🔧 Key Features
        - Real-time fraud detection
        - Batch processing capabilities
        - Probability scoring for risk assessment
        - Comprehensive model evaluation metrics
        - Interactive web interface
        - Professional code structure and documentation
        
        ### ⚠️ Demo Notice
        This is a demonstration version showing the system's interface and capabilities.
        The actual implementation includes trained ML models and full functionality.
        
        ### 📞 Contact
        **Brahimi Roumaissa**  
        Email: brahimiroumaissa1@gmail.com  
        GitHub: [Your GitHub Profile]
        """)

if __name__ == "__main__":
    main()
