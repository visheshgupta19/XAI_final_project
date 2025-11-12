"""
HELOC Credit Decision Dashboard
Interactive Streamlit app for credit decisions and improvement recommendations

To run: streamlit run heloc_dashboard.py

Prerequisites:
- Save your trained xgb model: import pickle; pickle.dump(xgb, open('xgb_model.pkl', 'wb'))
- Save test data: X_test.to_csv('X_test.csv', index=False)
- Save predictions: pd.DataFrame({'y_test': y_test, 'y_prob': y_prob}).to_csv('predictions.csv', index=False)
"""

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

st.set_page_config(
    page_title="HELOC Credit Decision System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "🏠 Dashboard Overview",
        "👤 Individual Assessment",
        "🎯 Recommendation Engine",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")

# Threshold slider
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.29,
    step=0.01,
    help="Probability threshold for approval (your optimal: 0.29)",
)

# Business parameters
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
        return None, None  # Already approved

    gap = threshold - current_prob
    improvement_options = []

    for feat_name in feature_names:
        feat_idx = feature_names.index(feat_name)
        current_val = instance[feat_idx]

        # Test improvement
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

    # Calculate metrics
    metrics = calculate_business_metrics(threshold)

    # Top metrics row
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

        # Create confusion matrix heatmap
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

        # Breakdown
        st.markdown(
            f"""
        - **True Positives:** {metrics['tp']} (Correctly approved)
        - **False Positives:** {metrics['fp']} (Bad customers approved) 💸
        - **False Negatives:** {metrics['fn']} (Good customers rejected) 😞
        - **True Negatives:** {metrics['tn']} (Correctly denied)
        """
        )

    with col2:
        st.subheader("Threshold Sensitivity")

        # Test multiple thresholds
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

        # Mark current threshold
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

        # Optimal threshold
        optimal_idx = np.argmax(profits)
        optimal_threshold = test_thresholds[optimal_idx]
        st.info(
            f"**Optimal threshold for max profit:** {optimal_threshold:.2%} (${max(profits):,.0f})"
        )

    # Distribution of predictions
    st.subheader("Prediction Distribution")

    fig = go.Figure()

    # Good applicants
    fig.add_trace(
        go.Histogram(
            x=y_prob[y_test == 1],
            name="Actually Good",
            marker_color="green",
            opacity=0.6,
            nbinsx=50,
        )
    )

    # Bad applicants
    fig.add_trace(
        go.Histogram(
            x=y_prob[y_test == 0],
            name="Actually Bad",
            marker_color="red",
            opacity=0.6,
            nbinsx=50,
        )
    )

    # Threshold line
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

    # Applicant selector
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

    # Get applicant data
    applicant = X_test.iloc[applicant_id]
    prob = y_prob[applicant_id]
    actual = "Good" if y_test[applicant_id] == 1 else "Bad"
    decision = "APPROVED" if prob >= threshold else "DENIED"

    # Display decision
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

    # Feature values
    st.subheader("📋 Applicant Profile")

    col1, col2 = st.columns(2)

    with col1:
        # Show top features
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
        # SHAP explanation (simplified)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(applicant.values.reshape(1, -1))

        # Top contributing features
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

    # Show all features in expandable section
    with st.expander("View All Features"):
        all_features = pd.DataFrame(
            {"Feature": feature_names, "Value": applicant.values}
        )
        st.dataframe(all_features, hide_index=True, use_container_width=True)


elif page == "🎯 Recommendation Engine":
    st.title("🎯 Improvement Recommendation Engine")
    st.markdown(
        "For denied applicants, this shows exactly what they need to improve to get approved."
    )

    # Find denied applicants
    denied = np.where(y_prob < threshold)[0]

    st.info(f"Found {len(denied):,} denied applications")

    # Select applicant
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_idx = st.selectbox(
            "Select Denied Applicant",
            denied,
            format_func=lambda x: f"Applicant #{x} (Prob: {y_prob[x]:.1%}, Actual: {'Good' if y_test[x]==1 else 'Bad'})",
        )

    with col2:
        show_top_n = st.slider("Show Top N Recommendations", 3, 10, 5)

    # Generate recommendations
    gap, improvement_df = generate_improvement_plan(selected_idx, threshold)

    if gap is None:
        st.success("This applicant is already approved!")
    else:
        # Current status
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Probability", f"{y_prob[selected_idx]:.1%}")
        with col2:
            st.metric("Target Threshold", f"{threshold:.1%}")
        with col3:
            st.metric(
                "Gap to Close", f"{gap:.1%}", delta=f"{gap:.1%}", delta_color="inverse"
            )

        # Top recommendations
        st.subheader("Top Improvement Recommendations")

        top_recs = improvement_df.head(show_top_n)

        for idx, row in top_recs.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{idx+1}. {row['Feature']}**")
                    st.markdown(
                        f"Current: `{row['Current']:.1f}` → Target: `{row['Target']:.1f}` (Change: `+{row['Change']:.1f}`)"
                    )

                    if row["Approves"]:
                        st.success(
                            f"✅ Impact: **+{row['Impact']:.1%}** - THIS ALONE WOULD APPROVE! (New prob: {row['New_Prob']:.1%})"
                        )
                    else:
                        st.info(
                            f"Impact: **+{row['Impact']:.1%}** (New prob: {row['New_Prob']:.1%})"
                        )

                with col2:
                    # Progress bar
                    progress = min(row["Impact"] / gap, 1.0)
                    st.progress(progress)
                    st.caption(f"{progress*100:.0f}% of gap")

                st.markdown("---")

        # Multi-feature strategy
        st.subheader("Combined Strategy")

        num_features = st.slider("How many features to combine?", 2, 5, 2)

        # Test combined improvement
        top_features = improvement_df.head(num_features)
        instance = X_test.iloc[selected_idx].values.copy()

        for _, row in top_features.iterrows():
            feat_idx = feature_names.index(row["Feature"])
            instance[feat_idx] = row["Target"]

        combined_prob = xgb.predict_proba(instance.reshape(1, -1))[0, 1]
        combined_gain = combined_prob - y_prob[selected_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Improve These Features Together:**")
            for _, row in top_features.iterrows():
                st.markdown(
                    f"- {row['Feature']}: `{row['Current']:.1f}` → `{row['Target']:.1f}`"
                )

        with col2:
            st.metric("Combined Impact", f"+{combined_gain:.1%}")
            st.metric("New Probability", f"{combined_prob:.1%}")

            if combined_prob >= threshold:
                st.success("✅ **APPROVED!**")
            else:
                still_needed = threshold - combined_prob
                st.warning(f"Still need +{still_needed:.1%} more")

        # Visualization
        st.subheader("Impact Comparison")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=improvement_df.head(show_top_n)["Feature"],
                y=improvement_df.head(show_top_n)["Impact"],
                marker_color=[
                    "green" if x else "orange"
                    for x in improvement_df.head(show_top_n)["Approves"]
                ],
                text=improvement_df.head(show_top_n)["Impact"].apply(
                    lambda x: f"+{x:.1%}"
                ),
                textposition="outside",
            )
        )

        fig.add_hline(
            y=gap,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Gap to close ({gap:.1%})",
        )

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Probability Gain",
            title="Which improvements have the most impact?",
            showlegend=False,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>💳 HELOC Credit Decision System | Built with Streamlit</p>
    <p>Model Threshold: {:.2%} | Dataset: {:,} applications</p>
</div>
""".format(
        threshold, len(X_test)
    ),
    unsafe_allow_html=True,
)
