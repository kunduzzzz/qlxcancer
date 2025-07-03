import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(page_title= "Bone Metastasis Risk Calculator", layout="wide")
st.title("Bone Metastasis Risk Calculator")
st.markdown("""
**Clinical Utility**:This tool predicts the risk of bone metastasis in newly diagnosed prostate cancer patients.
""")

# Load model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'xgboost_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Sidebar input features
with st.sidebar:
    st.header("Patient Parameters")
    
    st.subheader("Tumor Characteristics")
    T = st.slider("T Stage", min_value=2, max_value=4, value=3, step=1,
                 help=" T2: Organ-confined, T3: Extracapsular extension, T4: Invasion of adjacent structures")
    N = st.slider("N Stage", min_value=0, max_value=1, value=1, step=1,
                 help="N0: No regional lymph node metastasis, N1: Regional lymph node metastasis")
    
    st.subheader("Laboratory Values")
    PSA_density = st.slider("PSA Density (ng/mL/mL)", min_value=0.0, max_value=50.0, value=0.1, step=0.1,
                          help="Serum PSA divided by prostate volume")
    ALP = st.slider("ALP (U/L)", min_value=0, max_value=1100, value=100, step=0.5,
                  help="Alkaline phosphatase level")
    
    st.subheader("Performance Status")
    ECOG_PS = st.slider("ECOG PS", min_value=0, max_value=4, value=1, step=1,
                      help="0: Asymptomatic, 1: Symptomatic but ambulatory, 2: <50% in bed, 3: >50% in bed, 4: Bedbound")

# Create input dataframe
input_data = pd.DataFrame({
    'N': [N],
    'ECOG_PS': [ECOG_PS],
    'ALP': [ALP],
    'T': [T],
    'PSA_density': [PSA_density]
})

# Prediction and results
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Calculate Risk"):
        # Probability prediction
        prob = model.predict_proba(input_data)[0][1]
        risk_level = "High" if prob >= 0.6 else "Intermediate" if prob >= 0.3 else "Low"
        
        # Display results
        st.subheader("Risk Assessment")
        st.metric(label="Probability", value=f"{prob:.1%}")
        
        st.markdown(f"""
        **Risk Category**: {risk_level}
        
        **Clinical Considerations**:
        - High risk: Consider advanced imaging and multidisciplinary evaluation
        - Intermediate risk: Standard treatment protocols
        - Low risk: Active surveillance may be appropriate
        """)
with col2:
    if 'prob' in locals():
        st.subheader("Risk Factors")
        st.markdown(f"""
        Key contributing factors:
        - T stage: {'Significant' if T > 2 else 'Limited'} impact
        - N status: {'Positive' if N > 0 else 'Negative'} nodes
        - PSA density: {'Elevated' if PSA_density > 0.15 else 'Normal'}
        - ALP: {'Elevated' if ALP > 130 else 'Normal'}
        - ECOG PS: {[
            'Asymptomatic (0)', 
            'Symptomatic but ambulatory (1)', 
            '<50% in bed (2)', 
            '>50% in bed (3)', 
            'Bedbound (4)'
        ][ECOG_PS]}
        """)
        # Visual indicator with clinical thresholds
        risk_factors = {
            'T Stage': T/4,
            'N Status': N/3,
            'PSA Density': min(PSA_density/0.15, 1),  # 0.5作为PSA密度临床显著阈值
            'ALP': min(ALP/130, 1),  # 300作为ALP显著升高阈值
            'ECOG PS': ECOG_PS/4
        }
        st.bar_chart(pd.DataFrame.from_dict(
            {'Risk Contribution': risk_factors}), 
            height=300
        )


# Guidelines and references
st.markdown("---")
st.info("""
**Clinical Guidelines**:
- High risk: Bone scan + pelvic MRI recommended
- NCCN risk stratification applies
- Consider germline testing for high-risk patients
""")

# Disclaimer
st.warning("""
**Disclaimer**:
1. For clinical decision support only
2. Requires validation with clinical assessment
3. Not for diagnostic purposes
""")