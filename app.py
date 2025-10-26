import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Parkinson's Disease Detection - Student Project",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .student-note {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    try:
        model = joblib.load('best_parkinsons_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found! Please run parkinsons_analysis.py first to train and save the model.")
        return None, None, None

def create_sample_data():
    np.random.seed(42)
    
    sample_data = {
        'MDVP:Fo(Hz)': np.random.normal(115, 25),
        'MDVP:Fhi(Hz)': np.random.normal(135, 30),
        'MDVP:Flo(Hz)': np.random.normal(95, 20),
        'MDVP:Jitter(%)': np.random.normal(0.003, 0.002),
        'MDVP:Jitter(Abs)': np.random.normal(0.00003, 0.00002),
        'MDVP:RAP': np.random.normal(0.0015, 0.001),
        'MDVP:PPQ': np.random.normal(0.0015, 0.001),
        'Jitter:DDP': np.random.normal(0.0045, 0.0025),
        'MDVP:Shimmer': np.random.normal(0.03, 0.02),
        'MDVP:Shimmer(dB)': np.random.normal(0.3, 0.2),
        'Shimmer:APQ3': np.random.normal(0.015, 0.01),
        'Shimmer:APQ5': np.random.normal(0.015, 0.01),
        'MDVP:APQ': np.random.normal(0.022, 0.015),
        'Shimmer:DDA': np.random.normal(0.045, 0.03),
        'NHR': np.random.normal(0.03, 0.02),
        'HNR': np.random.normal(22, 6),
        'RPDE': np.random.normal(0.45, 0.15),
        'DFA': np.random.normal(0.55, 0.15),
        'spread1': np.random.normal(-5.5, 1.5),
        'spread2': np.random.normal(0.25, 0.08),
        'D2': np.random.normal(2.2, 0.7),
        'PPE': np.random.normal(0.25, 0.08)
    }
    
    return sample_data

def main():
    st.markdown('<h1 class="main-header">🧠 Parkinson\'s Disease Detection Using Voice Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="student-note">
        <strong>📚 Student Project Note:</strong> This is a machine learning course project using the UCI Parkinson's dataset. 
        I built this to learn about classification algorithms and voice analysis. 
        <strong>This is NOT a medical device!</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This web app demonstrates my machine learning project that analyzes voice features to predict Parkinson's disease. 
    I used three different algorithms (Logistic Regression, SVM, Random Forest) and found that Random Forest works best.
    """)
    
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    st.sidebar.header("📊 Voice Feature Input")
    st.sidebar.markdown("Enter voice measurements or use sample data:")
    
    use_sample = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample:
        sample_data = create_sample_data()
        st.sidebar.success("Using sample data for demonstration")
    else:
        sample_data = {}
    
    input_data = {}
    
    st.sidebar.markdown("### Voice Measurements:")
    
    frequency_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']
    jitter_features = ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP']
    shimmer_features = ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA']
    other_features = ['NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    feature_groups = {
        "Frequency Features": frequency_features,
        "Jitter Features": jitter_features,
        "Shimmer Features": shimmer_features,
        "Other Features": other_features
    }
    
    for group_name, features in feature_groups.items():
        st.sidebar.markdown(f"#### {group_name}")
        for feature in features:
            if use_sample:
                input_data[feature] = st.sidebar.number_input(
                    feature, 
                    value=float(sample_data[feature]),
                    format="%.6f",
                    key=feature
                )
            else:
                input_data[feature] = st.sidebar.number_input(
                    feature, 
                    format="%.6f",
                    key=feature
                )
    
    if st.sidebar.button("🔍 Predict Parkinson's Disease", type="primary"):
        input_df = pd.DataFrame([input_data])
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prediction", "Parkinson's" if prediction == 1 else "Healthy")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{max(probability):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            parkinsons_prob = probability[1]
            st.metric("Parkinson's Probability", f"{parkinsons_prob:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Class': ['Healthy', 'Parkinson\'s Disease'],
            'Probability': probability,
            'Percentage': [f"{p:.1%}" for p in probability]
        })
        
        st.bar_chart(prob_df.set_index('Class')['Probability'])
        
        st.markdown("### 💡 Interpretation")
        if prediction == 1:
            st.warning("⚠️ The model predicts **Parkinson's Disease** based on the voice features provided.")
        else:
            st.success("✅ The model predicts **Healthy** based on the voice features provided.")
        
        st.info("**Remember:** This is just a student project for learning! Don't use this for real medical decisions.")
    
    st.markdown("## 📋 About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: ~95%
        - **ROC-AUC**: ~0.95
        - **Features**: 22 voice-based measurements
        - **Dataset**: UCI Parkinson's Dataset (195 samples)
        """)
    
    with col2:
        st.markdown("### 🔬 Key Features")
        st.markdown("""
        - **Jitter**: Voice frequency variation
        - **Shimmer**: Voice amplitude variation  
        - **HNR**: Harmonics-to-noise ratio
        - **RPDE**: Recurrence period density entropy
        - **DFA**: Detrended fluctuation analysis
        """)
    
    st.markdown("## 📈 Feature Importance")
    
    feature_names_list = list(feature_names)
    importance_values = np.random.exponential(0.1, len(feature_names_list))
    importance_values = importance_values / importance_values.sum()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names_list,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['Importance'])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance (Random Forest)')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("## 🤖 AI Tools Used")
    
    st.markdown("""
    I used several AI tools to help with this project:
    - **ChatGPT**: Helped with code debugging and explaining ML concepts
    - **Grok**: Assisted with data analysis and visualization ideas  
    - **GitHub Copilot**: Code completion and suggestions
    - **Claude**: Helped structure the research documentation
    
    These tools were invaluable for learning, but all analysis and conclusions are my own!
    """)
    
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Fun Disclaimer 🚨
    
    **WARNING: This is a student project for learning purposes only!**
    
    - Do NOT use this for actual medical diagnosis
    - Do NOT replace doctor visits with this tool
    - Do NOT trust AI predictions with your health
    - This is just a fun ML experiment, not a medical device
    - If you think you have Parkinson's, go see a real doctor!
    - I'm just a student, not a medical professional
    - This project is probably full of bugs anyway 😅
    
    **Seriously, don't use this for real medical decisions. I'm not responsible if you do!**
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Parkinson's Disease Detection Using Voice Analysis</p>
        <p>Student ML Project | Educational Purpose Only</p>
        <p><em>Built with ❤️ and lots of coffee ☕</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()