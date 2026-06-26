import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Drug Risk Quantification System", layout="wide")

# --- TITLE & OVERVIEW ---
st.title("🛡️ Drug Risk Quantification & Capital Reserve System")
st.markdown("This system translates real-world clinical safety risks into quantifiable financial metrics for insurance underwriting and reserve management.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Simulation Controls")
theta_D = st.sidebar.slider("Drug Relative Risk Factor (Theta)", min_value=1.0, max_value=3.0, value=1.40, step=0.10)
st.sidebar.markdown("""
* **1.0**: Baseline standard population risk.
* **1.4**: Target drug increases mortality risk by 40%.
""")

# --- LOAD DATA (Your Cleaned SRS Table) ---
raw_text_data = """
Below 1 17.5 18.0 17.0 19.9 20.5 19.1 12.9 12.6 13.1
1-4 0.8 0.9 0.7 1.0 1.2 0.8 0.4 0.4 0.5
0-4 4.6 4.8 4.5 5.4 5.7 5.1 3.2 3.0 3.3
5-9 0.5 0.4 0.5 0.5 0.5 0.5 0.4 0.4 0.4
10-14 0.6 0.7 0.4 0.7 0.8 0.5 0.5 0.6 0.3
15-19 0.9 1.0 0.8 0.9 1.1 0.7 0.9 0.9 1.0
20-24 0.8 1.1 0.5 1.1 1.4 0.7 0.3 0.5 0.2
25-29 1.1 1.4 0.9 1.4 1.6 1.1 0.8 0.9 0.6
"""

rows = []
for line in raw_text_data.strip().split('\n'):
    parts = line.split()
    if parts[0] == "Below":
        age_group = "Below 1"
        metrics = parts[2:]
    else:
        age_group = parts[0]
        metrics = parts[1:]
    rows.append([age_group] + [float(m) for m in metrics])

columns = ['Age_Group', 'Total_Combined', 'Total_Male', 'Total_Female', 'Rural_Combined', 'Rural_Male', 'Rural_Female', 'Urban_Combined', 'Urban_Male', 'Urban_Female']
df_srs = pd.DataFrame(rows, columns=columns)

# --- PORTFOLIO SETUP ---
portfolio_data = {
    'Age_Group': ['Below 1', '1-4', '5-9', '10-14', '15-19', '20-24', '25-29'],
    'Patient_Count': [500, 2000, 3500, 5000, 8000, 12000, 11000],
    'Avg_Treatment_Cost_INR': [25000, 15000, 12000, 18000, 35000, 55000, 60000]
}
df_model = pd.merge(pd.DataFrame(portfolio_data), df_srs, on='Age_Group')

# --- ACTUARIAL MATH ---
df_model['qx_baseline'] = df_model['Total_Combined'] / 1000
df_model['qx_drug_adjusted'] = 1 - (1 - df_model['qx_baseline']) ** theta_D
df_model['Expected_Loss_INR'] = df_model['Patient_Count'] * df_model['qx_drug_adjusted'] * df_model['Avg_Treatment_Cost_INR']

# --- MACHINE LEARNING CLUSTERING ---
X = df_model[['qx_drug_adjusted', 'Expected_Loss_INR']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_model['Risk_Segment'] = kmeans.fit_predict(X)
cluster_mapping = {0: "Low Risk/Cost", 1: "High Risk/Cost", 2: "Medium Risk/Cost"}
df_model['Risk_Label'] = df_model['Risk_Segment'].map(cluster_mapping)

# --- DASHBOARD METRICS ---
total_reserve = df_model['Expected_Loss_INR'].sum()
max_impact_age = df_model.loc[df_model['Expected_Loss_INR'].idxmax()]['Age_Group']

col1, col2 = st.columns(2)
with col1:
    st.metric(label="💰 Total Required Capital Reserves", value=f"₹ {total_reserve:,.2f}")
with col2:
    st.metric(label="⚠️ Highest Risk Cohort (Age Group)", value=max_impact_age)

st.write("---")

# --- INTERACTIVE VISUALIZATIONS ---
st.subheader("Financial Impact Analysis Across Cohorts")
fig_bar = px.bar(df_model, x="Age_Group", y="Expected_Loss_INR", color="Risk_Label",
                 title="Expected Financial Loss & Risk Segmentation by Age Group",
                 labels={"Expected_Loss_INR": "Expected Loss (INR)", "Age_Group": "Age Group", "Risk_Label": "AI Risk Segment"},
                 color_discrete_map={"Low Risk/Cost": "#2ca02c", "Medium Risk/Cost": "#ff7f0e", "High Risk/Cost": "#d62728"})
st.plotly_chart(fig_bar, use_container_width=True)

# --- DATA TABLE VIEW ---
st.subheader("📋 Underlying Model Ledger Data")
st.dataframe(df_model[['Age_Group', 'Patient_Count', 'qx_baseline', 'qx_drug_adjusted', 'Expected_Loss_INR', 'Risk_Label']].style.format({
    'qx_baseline': '{:.4f}',
    'qx_drug_adjusted': '{:.4f}',
    'Expected_Loss_INR': '₹{:,.2f}'
}), use_container_width=True)
