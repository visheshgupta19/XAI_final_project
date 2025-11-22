import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px


def render_new_applicant_page():

    st.title("🆕 New Applicant Credit Check")
    st.markdown(
        """
    Enter your credit information below to:
    - Get instant approval/denial prediction
    - See what's helping or hurting your application
    - Get personalized recommendations to improve your chances
    """
    )

    # Feature definitions for tooltips
    feature_descriptions = {
        "ExternalRiskEstimate": "Consolidated risk marker (0-100, higher is better)",
        "MSinceOldestTradeOpen": "Months since oldest trade opened",
        "MSinceMostRecentTradeOpen": "Months since most recent trade opened",
        "AverageMInFile": "Average months in file",
        "NumSatisfactoryTrades": "Number of satisfactory trades",
        "NumTrades60Ever2DerogPubRec": "Number of trades 60+ days delinquent",
        "NumTrades90Ever2DerogPubRec": "Number of trades 90+ days delinquent",
        "PercentTradesNeverDelq": "Percent of trades never delinquent (0-100)",
        "MSinceMostRecentDelq": "Months since most recent delinquency",
        "MaxDelq2PublicRecLast12M": "Max delinquency in last 12 months (scale 1-9)",
        "MaxDelqEver": "Max delinquency ever (scale 1-9)",
        "NumTotalTrades": "Total number of trades",
        "NumTradesOpeninLast12M": "Number of trades opened in last 12 months",
        "PercentInstallTrades": "Percent of installment trades (0-100)",
        "MSinceMostRecentInqexcl7days": "Months since most recent inquiry (excluding last 7 days)",
        "NumInqLast6M": "Number of inquiries in last 6 months",
        "NumInqLast6Mexcl7days": "Number of inquiries in last 6 months (excluding last 7 days)",
        "NetFractionRevolvingBurden": "Revolving balance divided by credit limit (%)",
        "NetFractionInstallBurden": "Installment balance divided by original loan amount (%)",
        "NumRevolvingTradesWBalance": "Number of revolving trades with balance",
        "NumInstallTradesWBalance": "Number of installment trades with balance",
        "NumBank2NatlTradesWHighUtilization": "Number of bank/national trades with high utilization",
        "PercentTradesWBalance": "Percent of trades with balance (0-100)",
    }

    tab1, tab2, tab3 = st.tabs(["Manual Entry", "Upload Data", "Try Example"])

    with tab1:
        st.subheader("Enter Your Credit Information")

        # Create input form with organized sections
        with st.form("applicant_form"):
            st.markdown("### Basic Credit Profile")
            col1, col2, col3 = st.columns(3)

            with col1:
                external_risk = st.number_input(
                    "External Risk Score*",
                    min_value=0,
                    max_value=100,
                    value=70,
                    help=feature_descriptions["ExternalRiskEstimate"],
                )

                months_oldest_trade = st.number_input(
                    "Months Since Oldest Trade*",
                    min_value=0,
                    max_value=800,
                    value=180,
                    help=feature_descriptions["MSinceOldestTradeOpen"],
                )

                months_recent_trade = st.number_input(
                    "Months Since Recent Trade*",
                    min_value=0,
                    max_value=400,
                    value=5,
                    help=feature_descriptions["MSinceMostRecentTradeOpen"],
                )

            with col2:
                avg_months = st.number_input(
                    "Average Months in File*",
                    min_value=0,
                    max_value=400,
                    value=75,
                    help=feature_descriptions["AverageMInFile"],
                )

                num_satisfactory = st.number_input(
                    "Satisfactory Trades*",
                    min_value=0,
                    max_value=100,
                    value=20,
                    help=feature_descriptions["NumSatisfactoryTrades"],
                )

                num_total_trades = st.number_input(
                    "Total Trades*",
                    min_value=0,
                    max_value=120,
                    value=22,
                    help=feature_descriptions["NumTotalTrades"],
                )

            with col3:
                pct_never_delq = st.slider(
                    "% Trades Never Delinquent*",
                    min_value=0,
                    max_value=100,
                    value=95,
                    help=feature_descriptions["PercentTradesNeverDelq"],
                )

                pct_trades_balance = st.slider(
                    "% Trades with Balance*",
                    min_value=0,
                    max_value=100,
                    value=65,
                    help=feature_descriptions["PercentTradesWBalance"],
                )

                pct_install = st.slider(
                    "% Installment Trades",
                    min_value=0,
                    max_value=100,
                    value=32,
                    help=feature_descriptions["PercentInstallTrades"],
                )

            st.markdown("### Delinquency History")
            col1, col2, col3 = st.columns(3)

            with col1:
                trades_60_delq = st.number_input(
                    "Trades 60+ Days Delinquent",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help=feature_descriptions["NumTrades60Ever2DerogPubRec"],
                )

                trades_90_delq = st.number_input(
                    "Trades 90+ Days Delinquent",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help=feature_descriptions["NumTrades90Ever2DerogPubRec"],
                )

            with col2:
                months_since_delq = st.number_input(
                    "Months Since Last Delinquency",
                    min_value=-1,
                    max_value=100,
                    value=-1,
                    help="Enter -1 if never delinquent, or months since last delinquency",
                )

                max_delq_12m = st.selectbox(
                    "Max Delinquency (12M)",
                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                    index=5,
                    help="Scale 1-9, where 9 is no delinquency",
                )

            with col3:
                max_delq_ever = st.selectbox(
                    "Max Delinquency (Ever)",
                    options=[1, 2, 3, 4, 5, 6, 7, 8],
                    index=5,
                    help="Scale 1-8, where 8 is no delinquency",
                )

            st.markdown("### Credit Utilization")
            col1, col2, col3 = st.columns(3)

            with col1:
                revolving_burden = st.slider(
                    "Revolving Credit Burden (%)*",
                    min_value=0,
                    max_value=100,
                    value=30,
                    help=feature_descriptions["NetFractionRevolvingBurden"],
                )

                num_revolving_balance = st.number_input(
                    "Revolving Trades w/ Balance",
                    min_value=0,
                    max_value=35,
                    value=3,
                    help=feature_descriptions["NumRevolvingTradesWBalance"],
                )

            with col2:
                install_burden = st.slider(
                    "Installment Burden (%)",
                    min_value=0,
                    max_value=100,
                    value=45,
                    help=feature_descriptions["NetFractionInstallBurden"],
                )

                num_install_balance = st.number_input(
                    "Installment Trades w/ Balance",
                    min_value=0,
                    max_value=25,
                    value=2,
                    help=feature_descriptions["NumInstallTradesWBalance"],
                )

            with col3:
                high_util_trades = st.number_input(
                    "High Utilization Trades",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help=feature_descriptions["NumBank2NatlTradesWHighUtilization"],
                )

            st.markdown("### Recent Activity")
            col1, col2, col3 = st.columns(3)

            with col1:
                trades_opened_12m = st.number_input(
                    "Trades Opened (Last 12M)",
                    min_value=0,
                    max_value=20,
                    value=1,
                    help=feature_descriptions["NumTradesOpeninLast12M"],
                )

            with col2:
                inquiries_6m = st.number_input(
                    "Inquiries (Last 6M)",
                    min_value=0,
                    max_value=70,
                    value=1,
                    help=feature_descriptions["NumInqLast6M"],
                )

                inquiries_6m_excl7d = st.number_input(
                    "Inquiries 6M (excl 7d)",
                    min_value=0,
                    max_value=70,
                    value=1,
                    help=feature_descriptions["NumInqLast6Mexcl7days"],
                )

            with col3:
                months_recent_inq = st.number_input(
                    "Months Since Recent Inquiry",
                    min_value=-1,
                    max_value=25,
                    value=0,
                    help="Enter -1 if no inquiries, or months since last inquiry",
                )

            submitted = st.form_submit_button(
                "🎯 Check My Eligibility", use_container_width=True
            )

        if submitted:
            applicant_data = {
                "ExternalRiskEstimate": external_risk,
                "MSinceOldestTradeOpen": months_oldest_trade,
                "MSinceMostRecentTradeOpen": months_recent_trade,
                "AverageMInFile": avg_months,
                "NumSatisfactoryTrades": num_satisfactory,
                "NumTrades60Ever2DerogPubRec": trades_60_delq,
                "NumTrades90Ever2DerogPubRec": trades_90_delq,
                "PercentTradesNeverDelq": pct_never_delq,
                "MSinceMostRecentDelq": months_since_delq,
                "MaxDelq2PublicRecLast12M": max_delq_12m,
                "MaxDelqEver": max_delq_ever,
                "NumTotalTrades": num_total_trades,
                "NumTradesOpeninLast12M": trades_opened_12m,
                "PercentInstallTrades": pct_install,
                "MSinceMostRecentInqexcl7days": months_recent_inq,
                "NumInqLast6M": inquiries_6m,
                "NumInqLast6Mexcl7days": inquiries_6m_excl7d,
                "NetFractionRevolvingBurden": revolving_burden,
                "NetFractionInstallBurden": install_burden,
                "NumRevolvingTradesWBalance": num_revolving_balance,
                "NumInstallTradesWBalance": num_install_balance,
                "NumBank2NatlTradesWHighUtilization": high_util_trades,
                "PercentTradesWBalance": pct_trades_balance,
            }

            # display results
            display_prediction_results(applicant_data)

    with tab2:
        st.subheader("Upload Your Credit Data")
        st.markdown(
            """
        Upload a CSV file with your credit information. The file should contain these columns:
        `ExternalRiskEstimate`, `MSinceOldestTradeOpen`, `MSinceMostRecentTradeOpen`, etc.
        """
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                applicant_df = pd.read_csv(uploaded_file)

                # columns
                missing_cols = set(feature_names) - set(applicant_df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    st.success(f"Loaded {len(applicant_df)} applicant(s)")

                    # allow user select which row to analyze
                    if len(applicant_df) > 1:
                        row_idx = st.selectbox(
                            "Select applicant to analyze",
                            range(len(applicant_df)),
                            format_func=lambda x: f"Applicant {x+1}",
                        )
                    else:
                        row_idx = 0

                    if st.button("Analyze This Applicant"):
                        applicant_data = applicant_df.iloc[row_idx].to_dict()
                        display_prediction_results(applicant_data)

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab3:
        st.subheader("Try Example Scenarios")

        # example applicants
        examples = {
            "Strong Applicant (Should be Approved)": {
                "ExternalRiskEstimate": 82,
                "MSinceOldestTradeOpen": 250,
                "MSinceMostRecentTradeOpen": 3,
                "AverageMInFile": 95,
                "NumSatisfactoryTrades": 28,
                "NumTrades60Ever2DerogPubRec": 0,
                "NumTrades90Ever2DerogPubRec": 0,
                "PercentTradesNeverDelq": 100,
                "MSinceMostRecentDelq": -1,
                "MaxDelq2PublicRecLast12M": 9,
                "MaxDelqEver": 8,
                "NumTotalTrades": 30,
                "NumTradesOpeninLast12M": 2,
                "PercentInstallTrades": 30,
                "MSinceMostRecentInqexcl7days": 1,
                "NumInqLast6M": 1,
                "NumInqLast6Mexcl7days": 1,
                "NetFractionRevolvingBurden": 15,
                "NetFractionInstallBurden": 35,
                "NumRevolvingTradesWBalance": 2,
                "NumInstallTradesWBalance": 2,
                "NumBank2NatlTradesWHighUtilization": 0,
                "PercentTradesWBalance": 50,
            },
            "Borderline Applicant (Close Call)": {
                "ExternalRiskEstimate": 68,
                "MSinceOldestTradeOpen": 150,
                "MSinceMostRecentTradeOpen": 5,
                "AverageMInFile": 72,
                "NumSatisfactoryTrades": 18,
                "NumTrades60Ever2DerogPubRec": 1,
                "NumTrades90Ever2DerogPubRec": 0,
                "PercentTradesNeverDelq": 88,
                "MSinceMostRecentDelq": 24,
                "MaxDelq2PublicRecLast12M": 6,
                "MaxDelqEver": 6,
                "NumTotalTrades": 20,
                "NumTradesOpeninLast12M": 1,
                "PercentInstallTrades": 35,
                "MSinceMostRecentInqexcl7days": 0,
                "NumInqLast6M": 2,
                "NumInqLast6Mexcl7days": 2,
                "NetFractionRevolvingBurden": 45,
                "NetFractionInstallBurden": 52,
                "NumRevolvingTradesWBalance": 4,
                "NumInstallTradesWBalance": 2,
                "NumBank2NatlTradesWHighUtilization": 1,
                "PercentTradesWBalance": 72,
            },
            "High Risk Applicant (Likely Denied)": {
                "ExternalRiskEstimate": 52,
                "MSinceOldestTradeOpen": 80,
                "MSinceMostRecentTradeOpen": 2,
                "AverageMInFile": 45,
                "NumSatisfactoryTrades": 8,
                "NumTrades60Ever2DerogPubRec": 4,
                "NumTrades90Ever2DerogPubRec": 2,
                "PercentTradesNeverDelq": 72,
                "MSinceMostRecentDelq": 8,
                "MaxDelq2PublicRecLast12M": 4,
                "MaxDelqEver": 4,
                "NumTotalTrades": 12,
                "NumTradesOpeninLast12M": 3,
                "PercentInstallTrades": 40,
                "MSinceMostRecentInqexcl7days": 0,
                "NumInqLast6M": 6,
                "NumInqLast6Mexcl7days": 6,
                "NetFractionRevolvingBurden": 78,
                "NetFractionInstallBurden": 85,
                "NumRevolvingTradesWBalance": 7,
                "NumInstallTradesWBalance": 4,
                "NumBank2NatlTradesWHighUtilization": 3,
                "PercentTradesWBalance": 92,
            },
        }

        selected_example = st.selectbox("Choose an example:", list(examples.keys()))

        if st.button("Analyze This Example", use_container_width=True):
            display_prediction_results(examples[selected_example])


def display_prediction_results(applicant_data):

    applicant_df = pd.DataFrame([applicant_data])
    applicant_df = applicant_df[feature_names]

    prob = xgb.predict_proba(applicant_df)[0, 1]
    decision = "APPROVED" if prob >= threshold else "DENIED"
    decision_color = "green" if prob >= threshold else "red"

    st.markdown("---")
    st.markdown("## Your Credit Decision")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(
            f"### <span style='color:{decision_color}'>{decision}</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.metric(
            "Approval Probability",
            f"{prob:.1%}",
            delta=f"{prob - threshold:.1%} vs threshold",
        )

    with col3:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "green" if prob >= threshold else "red"},
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## What's Impacting Your Decision?")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(applicant_df)

    shap_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Your_Value": applicant_df.iloc[0].values,
            "SHAP_Impact": shap_values[0],
            "Dataset_Avg": X_test.mean().values,
        }
    )
    shap_importance["Abs_Impact"] = shap_importance["SHAP_Impact"].abs()
    shap_importance = shap_importance.sort_values("SHAP_Impact", ascending=False)

    helping = shap_importance[shap_importance["SHAP_Impact"] > 0].head(5)
    hurting = shap_importance[shap_importance["SHAP_Impact"] < 0].head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Factors HELPING Your Application")
        if len(helping) > 0:
            for idx, row in helping.iterrows():
                with st.container():
                    st.markdown(f"**{row['Feature']}**")
                    st.markdown(
                        f"Your value: `{row['Your_Value']:.1f}` | Avg: `{row['Dataset_Avg']:.1f}`"
                    )
                    st.progress(
                        float(
                            min(row["SHAP_Impact"] / helping["SHAP_Impact"].max(), 1.0)
                        )
                    )
                    st.markdown(
                        f"Impact: <span style='color:green'>+{row['SHAP_Impact']:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
        else:
            st.info("No significantly positive factors found")

    with col2:
        st.markdown("### Factors HURTING Your Application")
        if len(hurting) > 0:
            for idx, row in hurting.iterrows():
                with st.container():
                    st.markdown(f"**{row['Feature']}**")
                    st.markdown(
                        f"Your value: `{row['Your_Value']:.1f}` | Avg: `{row['Dataset_Avg']:.1f}`"
                    )
                    st.progress(
                        float(
                            min(
                                abs(row["SHAP_Impact"])
                                / hurting["SHAP_Impact"].abs().max(),
                                1.0,
                            )
                        )
                    )
                    st.markdown(
                        f"Impact: <span style='color:red'>{row['SHAP_Impact']:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
        else:
            st.info("No significantly negative factors found")

    if prob < threshold:
        st.markdown("## How to Improve Your Chances")

        gap = threshold - prob
        st.info(
            f"You need to improve your probability by **{gap:.1%}** to get approved"
        )

        improvement_options = []

        for feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            current_val = applicant_df.iloc[0, feat_idx]

            improvement = np.std(X_test[feat_name])
            new_instance = applicant_df.copy()
            new_instance.iloc[0, feat_idx] = current_val + improvement

            new_prob = xgb.predict_proba(new_instance)[0, 1]
            prob_gain = new_prob - prob

            if prob_gain > 0:
                improvement_options.append(
                    {
                        "Feature": feat_name,
                        "Current": current_val,
                        "Target": current_val + improvement,
                        "Change": improvement,
                        "Impact": prob_gain,
                        "New_Prob": new_prob,
                        "Approves": new_prob >= threshold,
                    }
                )

        improvement_df = pd.DataFrame(improvement_options).sort_values(
            "Impact", ascending=False
        )

        st.markdown("### Top 5 Recommendations")

        for i, (idx, row) in enumerate(improvement_df.head(5).iterrows(), 1):
            with st.expander(
                f"#{i}: Improve **{row['Feature']}** (Impact: +{row['Impact']:.1%})",
                expanded=(i <= 2),
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Current Value:** {row['Current']:.1f}")
                    st.markdown(f"**Target Value:** {row['Target']:.1f}")
                    st.markdown(f"**Change Needed:** +{row['Change']:.1f}")
                    st.markdown(f"**New Probability:** {row['New_Prob']:.1%}")

                    if row["Approves"]:
                        st.success("**THIS ALONE WOULD GET YOU APPROVED!**")
                    else:
                        remaining = threshold - row["New_Prob"]
                        st.warning(f"Would still need +{remaining:.1%} more")

                with col2:
                    progress = min(row["Impact"] / gap, 1.0)
                    st.metric("Closes Gap", f"{progress*100:.0f}%")
                    st.progress(float(progress))

        st.markdown("### Combined Improvement Strategy")

        num_features = st.slider("How many factors to improve together?", 2, 5, 3)

        top_features = improvement_df.head(num_features)
        combined_instance = applicant_df.copy()

        for _, row in top_features.iterrows():
            feat_idx = feature_names.index(row["Feature"])
            combined_instance.iloc[0, feat_idx] = row["Target"]

        combined_prob = xgb.predict_proba(combined_instance)[0, 1]
        combined_gain = combined_prob - prob

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**If you improve these together:**")
            for _, row in top_features.iterrows():
                st.markdown(
                    f"- **{row['Feature']}**: {row['Current']:.1f} → {row['Target']:.1f}"
                )

        with col2:
            st.metric("Combined Impact", f"+{combined_gain:.1%}")
            st.metric("New Probability", f"{combined_prob:.1%}")

            if combined_prob >= threshold:
                st.success("**YOU'D BE APPROVED!**")
            else:
                still_needed = threshold - combined_prob
                st.warning(f"Need +{still_needed:.1%} more")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=improvement_df.head(8)["Feature"],
                y=improvement_df.head(8)["Impact"] * 100,
                marker_color=[
                    "green" if x else "orange"
                    for x in improvement_df.head(8)["Approves"]
                ],
                text=improvement_df.head(8)["Impact"].apply(lambda x: f"+{x:.1%}"),
                textposition="outside",
            )
        )

        fig.add_hline(
            y=gap * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Gap to close ({gap:.1%})",
        )

        fig.update_layout(
            xaxis_title="Feature to Improve",
            yaxis_title="Probability Gain (%)",
            title="Which improvements have the biggest impact?",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.success("## Congratulations! You're Approved!")
        st.markdown(
            """
        Your application meets our criteria. Here's what's working in your favor:
        """
        )

        st.markdown("### Your Strongest Factors:")
        for idx, row in helping.head(3).iterrows():
            st.markdown(
                f"**{row['Feature']}**: {row['Your_Value']:.1f} (Impact: +{row['SHAP_Impact']:.3f})"
            )

    st.markdown("---")
    if st.button("Download Full Report (CSV)", use_container_width=True):
        report_df = pd.DataFrame(
            {
                "Metric": ["Decision", "Probability", "Threshold"],
                "Value": [decision, f"{prob:.1%}", f"{threshold:.1%}"],
            }
        )

        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Report",
            data=csv,
            file_name="credit_decision_report.csv",
            mime="text/csv",
        )


st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
    }
    .denied {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'xgb_model.pkl' not found. Please train and save your model first."
        )
        st.stop()


@st.cache_data
def load_data():
    """Load test data and predictions"""
    try:
        X_test = pd.read_csv("X_test.csv")
        predictions = pd.read_csv("predictions.csv")
        return X_test, predictions
    except FileNotFoundError:
        st.error("Data files not found. Please save X_test.csv and predictions.csv")
        st.stop()


xgb = load_model()
X_test, predictions = load_data()
y_test = predictions["y_test"].values
y_prob = predictions["y_prob"].values
feature_names = X_test.columns.tolist()

st.sidebar.title("🏦 HELOC Credit System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🆕 New Applicant Check",
        "🏠 Dashboard Overview",
        "👤 Individual Assessment",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")

# slider
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.29,
    step=0.01,
    help="Probability threshold for approval (your optimal: 0.29)",
)

st.sidebar.subheader("Business Parameters")
profit_per_approval = st.sidebar.number_input(
    "Profit per Good Customer ($)", min_value=0, value=1000, step=100
)

default_cost = st.sidebar.number_input(
    "Cost per Default ($)", min_value=0, value=10000, step=500
)


def calculate_business_metrics(threshold):
    """Calculate business metrics at given threshold"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_approvals = tp + fp
    approval_rate = total_approvals / len(y_test)
    revenue = tp * profit_per_approval
    losses = fp * default_cost
    net_profit = revenue - losses

    return {
        "approval_rate": approval_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "net_profit": net_profit,
        "revenue": revenue,
        "losses": losses,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def generate_improvement_plan(applicant_idx, threshold):
    """Generate improvement recommendations for an applicant"""
    instance = X_test.iloc[applicant_idx].values
    current_prob = y_prob[applicant_idx]

    if current_prob >= threshold:
        return None, None

    gap = threshold - current_prob
    improvement_options = []

    for feat_name in feature_names:
        feat_idx = feature_names.index(feat_name)
        current_val = instance[feat_idx]

        #  improvement
        improvement = np.std(X_test[feat_name])
        new_instance = instance.copy()
        new_instance[feat_idx] = current_val + improvement

        new_prob = xgb.predict_proba(new_instance.reshape(1, -1))[0, 1]
        prob_gain = new_prob - current_prob

        improvement_options.append(
            {
                "Feature": feat_name,
                "Current": current_val,
                "Target": current_val + improvement,
                "Change": improvement,
                "Impact": prob_gain,
                "New_Prob": new_prob,
                "Approves": new_prob >= threshold,
            }
        )

    improvement_df = pd.DataFrame(improvement_options).sort_values(
        "Impact", ascending=False
    )
    return gap, improvement_df


if page == "🏠 Dashboard Overview":
    st.title("🏠 Credit Decision System Overview")
    st.markdown(
        f"**Current Threshold:** {threshold:.2%} | **Dataset Size:** {len(X_test):,} applications"
    )

    metrics = calculate_business_metrics(threshold)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Approval Rate",
            f"{metrics['approval_rate']:.1%}",
            help="Percentage of applications approved",
        )

    with col2:
        st.metric(
            "Net Profit",
            f"${metrics['net_profit']:,.0f}",
            help="Revenue minus default losses",
        )

    with col3:
        st.metric(
            "Precision",
            f"{metrics['precision']:.1%}",
            help="% of approvals that are actually good",
        )

    with col4:
        st.metric(
            "Recall", f"{metrics['recall']:.1%}", help="% of good applicants we approve"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        # confusion matrix heatmap
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Bad", "Good"],
            yticklabels=["Bad", "Good"],
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

        st.markdown(
            f"""
        - **True Positives:** {metrics['tp']} (Correctly approved)
        - **False Positives:** {metrics['fp']} (Bad customers approved) 
        - **False Negatives:** {metrics['fn']} (Good customers rejected) 
        - **True Negatives:** {metrics['tn']} (Correctly denied)
        """
        )

    with col2:
        st.subheader("Threshold Sensitivity")

        test_thresholds = np.linspace(0.1, 0.9, 50)
        threshold_results = [calculate_business_metrics(t) for t in test_thresholds]

        profits = [r["net_profit"] for r in threshold_results]
        approval_rates = [r["approval_rate"] for r in threshold_results]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=test_thresholds,
                y=profits,
                mode="lines",
                name="Net Profit",
                line=dict(color="green", width=3),
            )
        )

        # current threshold
        current_metrics = calculate_business_metrics(threshold)
        fig.add_trace(
            go.Scatter(
                x=[threshold],
                y=[current_metrics["net_profit"]],
                mode="markers",
                name="Current",
                marker=dict(size=15, color="red"),
            )
        )

        fig.update_layout(
            title="Profit vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Net Profit ($)",
            hovermode="x",
        )

        st.plotly_chart(fig, use_container_width=True)

        optimal_idx = np.argmax(profits)
        optimal_threshold = test_thresholds[optimal_idx]
        st.info(
            f"**Optimal threshold for max profit:** {optimal_threshold:.2%} (${max(profits):,.0f})"
        )

    st.subheader("Prediction Distribution")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=y_prob[y_test == 1],
            name="Actually Good",
            marker_color="green",
            opacity=0.6,
            nbinsx=50,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=y_prob[y_test == 0],
            name="Actually Bad",
            marker_color="red",
            opacity=0.6,
            nbinsx=50,
        )
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold ({threshold:.2%})",
    )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Predicted Probability (Good)",
        yaxis_title="Count",
        title="How well does the model separate Good vs Bad?",
        hovermode="x",
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "👤 Individual Assessment":
    st.title("👤 Individual Credit Assessment")

    # selector
    col1, col2 = st.columns([2, 1])

    with col1:
        applicant_id = st.number_input(
            "Enter Applicant ID (0 to {}):".format(len(X_test) - 1),
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
        )

    with col2:
        filter_type = st.selectbox(
            "Quick Filter",
            ["All", "Borderline (40-60%)", "High Risk (<30%)", "Low Risk (>70%)"],
        )

        if filter_type == "Borderline (40-60%)":
            filtered = np.where((y_prob >= 0.4) & (y_prob <= 0.6))[0]
        elif filter_type == "High Risk (<30%)":
            filtered = np.where(y_prob < 0.3)[0]
        elif filter_type == "Low Risk (>70%)":
            filtered = np.where(y_prob > 0.7)[0]
        else:
            filtered = np.arange(len(X_test))

        if len(filtered) > 0 and st.button("Show Random from Filter"):
            applicant_id = np.random.choice(filtered)

    # get data
    applicant = X_test.iloc[applicant_id]
    prob = y_prob[applicant_id]
    actual = "Good" if y_test[applicant_id] == 1 else "Bad"
    decision = "APPROVED" if prob >= threshold else "DENIED"

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"### Decision: <span class='{decision.lower()}'>{decision}</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Predicted Probability", f"{prob:.1%}")

    with col3:
        st.metric(
            "Actual Outcome",
            actual,
            delta=(
                "Correct" if (decision == "APPROVED") == (actual == "Good") else "Wrong"
            ),
            delta_color=(
                "normal"
                if (decision == "APPROVED") == (actual == "Good")
                else "inverse"
            ),
        )

    st.subheader("Applicant Profile")

    col1, col2 = st.columns(2)

    with col1:
        # top features
        top_features = [
            "ExternalRiskEstimate",
            "NetFractionRevolvingBurden",
            "PercentTradesNeverDelq",
            "PercentTradesWBalance",
        ]

        profile_data = []
        for feat in top_features:
            val = applicant[feat]
            mean = X_test[feat].mean()
            profile_data.append(
                {
                    "Feature": feat,
                    "Value": f"{val:.1f}",
                    "Dataset Avg": f"{mean:.1f}",
                    "vs Avg": f"{'+' if val > mean else ''}{val - mean:.1f}",
                }
            )

        st.dataframe(
            pd.DataFrame(profile_data), hide_index=True, use_container_width=True
        )

    with col2:
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(applicant.values.reshape(1, -1))

        shap_importance = (
            pd.DataFrame({"Feature": feature_names, "Impact": shap_values[0]})
            .sort_values("Impact", key=abs, ascending=False)
            .head(8)
        )

        fig = go.Figure(
            go.Bar(
                x=shap_importance["Impact"],
                y=shap_importance["Feature"],
                orientation="h",
                marker_color=[
                    "green" if x > 0 else "red" for x in shap_importance["Impact"]
                ],
            )
        )

        fig.update_layout(
            title="Why This Decision?",
            xaxis_title="Impact on Prediction",
            yaxis_title="",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View All Features"):
        all_features = pd.DataFrame(
            {"Feature": feature_names, "Value": applicant.values}
        )
        st.dataframe(all_features, hide_index=True, use_container_width=True)

elif page == "🆕 New Applicant Check":
    render_new_applicant_page()


st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>HELOC Credit Decision System | Built with Streamlit</p>
    <p>Model Threshold: {:.2%} | Dataset: {:,} applications</p>
</div>
""".format(
        threshold, len(X_test)
    ),
    unsafe_allow_html=True,
)


"""
The streamlit app has been created with the help of ChatGPT-5 when I ran into any error. 
This has been done over 1-2 weeks which is why I donot have the exact time stamps and prompt.
"""
