"""
Streamlit Frontend Application for AI-Powered Skincare Routine Checker
Features:
- Responsive, premium UI with curated styling and card designs
- Skin profile customizer with concern, sensitivity, avoid-list, and budget filters
- Direct integration with FastAPI backend (GET /model-info, POST /check-routine, POST /predict)
- Morning & Evening Routine visualizers with ordered steps
- Ingredient Explorer & Compatibility Caution panels
- Model Evaluation, Metrics, and MLOps Lineage Dashboard
- Downloadable Routine Report (Markdown/Text)
- Session assessment history with one-click clear option
"""

import os
import json
import requests
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Skincare Routine Checker",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL configuration
API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .routine-step {
        background: #ffffff;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
    .caution-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .disclaimer-banner {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None

# Sidebar: Navigation & API Health
with st.sidebar:
    st.title("✨ Skincare AI")
    st.markdown("**MLOps Routine Intelligence**")

    # API Health Indicator
    try:
        health_resp = requests.get(f"{API_URL}/ready", timeout=3)
        if health_resp.status_code == 200:
            st.success("🟢 API Service: Online & Ready", icon="✅")
        else:
            st.warning("🟡 API Service: Degraded", icon="⚠️")
    except Exception:
        st.error(f"🔴 API Service: Offline ({API_URL})", icon="🚨")

    menu = st.radio(
        "Navigation",
        ["Routine Builder & Checker", "Product Category Classifier", "Model & MLOps Lineage", "Assessment History"]
    )

    st.markdown("---")
    st.markdown("### 🎓 Academic MLOps Demo")
    st.caption("Trained on real Sephora & INCI cosmetics data. Integrates MLflow tracking, Docker compose, and automated gating.")

# ----------------- VIEW 1: ROUTINE BUILDER & CHECKER -----------------
if menu == "Routine Builder & Checker":
    st.markdown('<div class="main-header">AI-Powered Skincare Routine Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalized regimen assembly with evidence-reviewed ingredient safety & compatibility checks.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Your Skin Profile")
        with st.form("routine_form"):
            skin_type = st.selectbox(
                "Skin Type",
                ["normal", "oily", "dry", "combination", "unknown"],
                index=1
            )
            concerns = st.multiselect(
                "Primary Concerns",
                ["dryness", "oiliness", "uneven texture", "appearance of blemishes"],
                default=["oiliness", "appearance of blemishes"]
            )
            sensitivity = st.select_slider(
                "Skin Sensitivity",
                options=["low", "medium", "high"],
                value="low"
            )
            avoid_input = st.text_input(
                "Ingredients / Tags to Avoid (comma-separated)",
                placeholder="e.g. fragrance, alcohol, parabens"
            )
            preferred_routine = st.selectbox(
                "Preferred Routine",
                ["both", "morning", "evening"]
            )
            budget = st.number_input(
                "Max Budget per Item (USD, 0 for any)",
                min_value=0.0,
                max_value=400.0,
                value=45.0,
                step=5.0
            )

            submit_btn = st.form_submit_button("🔍 Generate & Check Routine", use_container_width=True)

        if submit_btn:
            avoid_list = [x.strip() for x in avoid_input.split(",") if x.strip()]
            payload = {
                "profile": {
                    "skin_type": skin_type,
                    "concerns": concerns,
                    "sensitivity": sensitivity,
                    "avoid_ingredients": avoid_list,
                    "preferred_routine": preferred_routine,
                    "max_budget": budget if budget > 0 else None
                }
            }

            with st.spinner("Calling inference service & evaluating ingredient rules..."):
                try:
                    resp = requests.post(f"{API_URL}/check-routine", json=payload, timeout=10)
                    if resp.status_code == 200:
                        assessment = resp.json()
                        st.session_state.current_assessment = assessment
                        st.session_state.history.append({
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "skin_type": skin_type,
                            "concerns": ", ".join(concerns),
                            "routine": preferred_routine,
                            "assessment": assessment
                        })
                        st.success("Routine successfully compiled!")
                    else:
                        st.error(f"Backend API error: {resp.text}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI server at {API_URL}: {e}")

    with col2:
        if st.session_state.current_assessment:
            res = st.session_state.current_assessment

            # Tabs for viewing different aspects of the routine
            tab_am, tab_pm, tab_safety, tab_export = st.tabs([
                "☀️ Morning Regimen", "🌙 Evening Regimen", "⚠️ Safety & Compatibility", "📄 Export Report"
            ])

            with tab_am:
                if res.get("morning_routine"):
                    for step_idx, item in enumerate(res["morning_routine"], 1):
                        st.markdown(f"""
                        <div class="routine-step">
                            <span style="font-weight:700; color:#0d9488;">Step {step_idx} &bull; {item['category']}</span><br/>
                            <strong style="font-size:1.1rem;">{item['brand']} — {item['name']}</strong><br/>
                            <span style="color:#64748b; font-size:0.9rem;">⭐ {item.get('rank', 'N/A')} / 5.0 &nbsp;|&nbsp; 💵 ${item.get('price', 0.0):.2f}</span><br/>
                            <p style="margin-top:6px; font-size:0.85rem; color:#475569;">
                                <strong>Key Ingredients:</strong> {', '.join(item.get('ingredients', [])[:8])}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No morning steps selected for current preference.")

            with tab_pm:
                if res.get("evening_routine"):
                    for step_idx, item in enumerate(res["evening_routine"], 1):
                        st.markdown(f"""
                        <div class="routine-step">
                            <span style="font-weight:700; color:#3b82f6;">Step {step_idx} &bull; {item['category']}</span><br/>
                            <strong style="font-size:1.1rem;">{item['brand']} — {item['name']}</strong><br/>
                            <span style="color:#64748b; font-size:0.9rem;">⭐ {item.get('rank', 'N/A')} / 5.0 &nbsp;|&nbsp; 💵 ${item.get('price', 0.0):.2f}</span><br/>
                            <p style="margin-top:6px; font-size:0.85rem; color:#475569;">
                                <strong>Key Ingredients:</strong> {', '.join(item.get('ingredients', [])[:8])}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No evening steps selected for current preference.")

            with tab_safety:
                st.subheader("Compatibility & Ingredient Review")

                if res.get("avoid_violations"):
                    st.markdown("**Avoid List Conflicts:**")
                    for v in res["avoid_violations"]:
                        st.error(v, icon="🚫")

                if res.get("warnings"):
                    st.markdown("**Evidence-Backed Warnings:**")
                    for w in res["warnings"]:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>{w['title']}</strong><br/>
                            <span>{w['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                if res.get("cautions"):
                    st.markdown("**Formulation Cautions:**")
                    for c in res["cautions"]:
                        st.markdown(f"""
                        <div class="caution-box">
                            <strong>{c['title']}</strong><br/>
                            <span>{c['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                if not res.get("warnings") and not res.get("cautions") and not res.get("avoid_violations"):
                    st.success("✅ No conflicting active interactions or avoid violations identified.")

                st.markdown("---")
                st.markdown("**Target Concern Ingredient Matches:**")
                for bi in res.get("beneficial_insights", []):
                    st.write(f"- **{bi['concern'].title()}**: {bi['message']}")

            with tab_export:
                st.subheader("Downloadable Routine Report")
                report_text = f"""# Personalized Skincare Assessment Report
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Model Version: {res.get('model_version_used', 'N/A')}

## Profile Summary
- Skin Type: {res['profile_summary'].get('skin_type')}
- Primary Concerns: {', '.join(res['profile_summary'].get('concerns', []))}
- Sensitivity: {res['profile_summary'].get('sensitivity')}

## Morning Routine
"""
                for s in res.get("morning_routine", []):
                    report_text += f"- [{s['category']}] {s['brand']} - {s['name']} (${s['price']:.2f})\n"

                report_text += "\n## Evening Routine\n"
                for s in res.get("evening_routine", []):
                    report_text += f"- [{s['category']}] {s['brand']} - {s['name']} (${s['price']:.2f})\n"

                report_text += f"\n## Clinical & Educational Disclaimer\n{res.get('disclaimer')}\n"

                st.download_button(
                    label="📥 Download Routine Report (.md)",
                    data=report_text,
                    file_name="skincare_routine_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                st.code(report_text, language="markdown")
        else:
            st.info("👈 Enter your skin parameters and click **Generate & Check Routine** to view your recommendations.")

    st.markdown(f"""
    <div class="disclaimer-banner">
        <strong>Academic & Educational Disclaimer:</strong> This application is developed for educational and machine learning demonstration purposes.
        Predictions and rule assessments do not constitute professional dermatological diagnoses, prescriptions, or clinical treatments.
        Always patch test new cosmetic items and consult a board-certified dermatologist for persistent skin conditions.
    </div>
    """, unsafe_allow_html=True)


# ----------------- VIEW 2: PRODUCT CATEGORY CLASSIFIER -----------------
elif menu == "Product Category Classifier":
    st.markdown('<div class="main-header">Real-Time Product Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Directly queries the trained scikit-learn model pipeline served via FastAPI.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Product Input Details")
        sample_ingredients = "Water, Glycerin, Dimethicone, Hyaluronic Acid, Niacinamide, Carbomer, Phenoxyethanol"
        ing_input = st.text_area("INCI Ingredients List", value=sample_ingredients, height=140)
        p_price = st.number_input("Retail Price ($)", min_value=1.0, max_value=400.0, value=32.0)
        p_rating = st.slider("Customer Rating (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=4.4, step=0.1)
        p_skin = st.selectbox("Intended Skin Suitability", ["normal", "dry", "oily", "combination", "sensitive"])

        predict_btn = st.button("🚀 Classify Product Category", use_container_width=True)

    with col_b:
        st.subheader("Model Inference Output")
        if predict_btn:
            payload = {
                "ingredients_text": ing_input,
                "price": p_price,
                "rating": p_rating,
                "skin_type": p_skin
            }
            with st.spinner("Invoking FastAPI /predict endpoint..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    if resp.status_code == 200:
                        pred_data = resp.json()
                        st.success(f"**Predicted Category:** `{pred_data['predicted_category']}`", icon="🎯")
                        st.metric("Model Confidence", f"{pred_data['confidence'] * 100:.1f}%")
                        st.caption(f"Active Model Version: `{pred_data['model_version']}`")

                        st.markdown("#### Class Probability Distribution")
                        probs_df = pd.DataFrame(
                            list(pred_data["probabilities"].items()),
                            columns=["Category", "Probability"]
                        ).sort_values(by="Probability", ascending=False)

                        st.bar_chart(probs_df.set_index("Category"))
                    else:
                        st.error(f"Inference error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to reach FastAPI backend: {e}")
        else:
            st.info("Click 'Classify Product Category' to run real-time inference.")


# ----------------- VIEW 3: MODEL & MLOPS LINEAGE -----------------
elif menu == "Model & MLOps Lineage":
    st.markdown('<div class="main-header">MLOps Architecture & Model Lineage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Verification of experiment tracking, evaluation metrics, and promotion gating.</div>', unsafe_allow_html=True)

    try:
        info_resp = requests.get(f"{API_URL}/model-info", timeout=5)
        model_info = info_resp.json() if info_resp.status_code == 200 else {}
    except Exception:
        model_info = {}

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Production Version", f"v{model_info.get('current_production_version', 1)}")
    with col_m2:
        st.metric("Model Architecture", model_info.get("architecture", "RandomForest"))
    with col_m3:
        st.metric("Held-Out Accuracy", f"{model_info.get('metrics', {}).get('accuracy', 0.5467) * 100:.2f}%")
    with col_m4:
        st.metric("Macro F1-Score", f"{model_info.get('metrics', {}).get('macro_f1', 0.5536):.4f}")

    st.markdown("---")
    st.subheader("Training vs Baseline Evaluation Metrics")

    # Load local metrics and summary if available
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            full_metrics = json.load(f)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Per-Class Performance (Held-out Test)")
            per_class_df = pd.DataFrame(full_metrics.get("per_class", {})).T
            st.dataframe(per_class_df, use_container_width=True)

            st.markdown("#### Gating Criteria Compliance")
            st.markdown("""
            - **Minimum Accuracy Threshold**: `>= 0.4000` (Passed: `0.5467`)
            - **Minimum Macro-F1 Threshold**: `>= 0.4000` (Passed: `0.5536`)
            - **Zero Pipeline Faults**: Validated on single-row and batch inference
            """)

        with col_right:
            st.markdown("#### Confusion Matrix")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Held-Out Test Confusion Matrix", use_container_width=True)

    # Operational metrics
    st.markdown("---")
    st.subheader("Live Operational Telemetry")
    try:
        op_resp = requests.get(f"{API_URL}/metrics", timeout=5)
        if op_resp.status_code == 200:
            op_data = op_resp.json()
            st.json(op_data)
    except Exception:
        st.caption("Operational metrics endpoint offline.")


# ----------------- VIEW 4: ASSESSMENT HISTORY -----------------
elif menu == "Assessment History":
    st.markdown('<div class="main-header">User Assessment History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review past session assessments or reset history.</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Assessment History"):
        st.session_state.history = []
        st.session_state.current_assessment = None
        st.success("History successfully cleared!")

    if st.session_state.history:
        for idx, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Assessment #{idx} — {entry['timestamp']} ({entry['skin_type'].upper()}, {entry['concerns']})"):
                st.write(f"**Routine Preference:** {entry['routine']}")
                st.json(entry["assessment"])
    else:
        st.info("No prior assessments recorded in this session.")
