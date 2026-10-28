"""
Streamlit Frontend Application — Skincare Routine Ingredient Checker
=====================================================================
Five primary sections:
1. Routine Conflict Checker  : Search/add products, detect active conflicts & duplicate actives.
2. Product Catalog Search    : Full-text & fuzzy search with ingredient provenance view.
3. Ingredient Analysis       : Paste raw list → parse → normalize synonyms → classify functions → show concentrations.
4. Packaging Label OCR       : Upload photo → extract text → edit → add to routine.
5. MLOps & Drift Telemetry   : Model versions, MLflow runs, evaluation metrics, drift indicators, latency.

All predictions come from the live FastAPI backend. No hardcoded example outputs.
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="Skincare Routine Ingredient Checker",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header-title {
    font-size: 2.05rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0369a1 0%, #0d9488 60%, #16a34a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.header-subtitle {
    color: #475569;
    font-size: 1.0rem;
    margin-bottom: 1.2rem;
}
.status-safe {
    background: #dcfce7; color: #166534;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 600; display: inline-block;
}
.status-conflict {
    background: #fee2e2; color: #991b1b;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 600; display: inline-block;
}
.status-duplicate {
    background: #fef9c3; color: #854d0e;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 600; display: inline-block;
}
.status-avoid {
    background: #fce7f3; color: #9d174d;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 600; display: inline-block;
}
.warning-card {
    background: #fff1f2; border-left: 4px solid #f43f5e;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
}
.caution-card {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
}
.info-card {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
}
.product-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.disclaimer-box {
    background: #f8fafc; border-radius: 8px;
    border: 1px solid #e2e8f0; padding: 14px;
    font-size: 0.83rem; color: #64748b; margin-top: 24px;
}

.synergy-card {
    background: #f0fdf4; border-left: 4px solid #10b981;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
}
.barrier-card-green {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 15px;
}
.barrier-card-amber {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 15px;
}
.barrier-card-red {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 15px;
}

.metric-block {
    background: #f0f9ff; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
    border-left: 3px solid #0284c7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Session State ───────────────────────────
if "routine_products" not in st.session_state:
    st.session_state.routine_products = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "ocr_text_buffer" not in st.session_state:
    st.session_state.ocr_text_buffer = ""
if "parsed_ingredient_results" not in st.session_state:
    st.session_state.parsed_ingredient_results = None

# ─────────────────────────── Helpers ───────────────────────────
def api_get(endpoint: str, params: dict = None, timeout: int = 5):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict, timeout: int = 10):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def render_status_badge(status: str):
    if status == "POTENTIAL_CONFLICT":
        st.markdown('<span class="status-conflict">⚠️ POTENTIAL CONFLICT DETECTED</span>', unsafe_allow_html=True)
    elif status == "DUPLICATE_ACTIVE":
        st.markdown('<span class="status-duplicate">🟡 DUPLICATE ACTIVE IDENTIFIED</span>', unsafe_allow_html=True)
    elif status == "USER_AVOID_LIST_MATCH":
        st.markdown('<span class="status-avoid">🚫 AVOID-LIST MATCH</span>', unsafe_allow_html=True)
    elif status == "INSUFFICIENT_INFORMATION":
        st.markdown('<span class="status-duplicate">ℹ️ INSUFFICIENT INFORMATION</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-safe">✅ NO CONFIGURED CONFLICT FOUND</span>', unsafe_allow_html=True)


def function_color(func: str) -> str:
    palette = {
        "Humectant": "🔵",
        "Emollient": "🟢",
        "Occlusive": "🟤",
        "Exfoliating agent": "🔴",
        "Antioxidant": "🟠",
        "Surfactant": "🟣",
        "Soothing / Anti-inflammatory": "🩵",
        "Sunscreen active": "🌕",
    }
    return palette.get(func, "⚪")


# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.title("🧴 Skincare Routine Checker")
    st.caption("ML-Powered Ingredient Analysis & MLOps")

    # API Status
    try:
        r_ready = requests.get(f"{API_URL}/ready", timeout=2)
        if r_ready.status_code == 200:
            st.success("🟢 API: Online", icon="✅")
        else:
            st.warning("🟡 API: Degraded", icon="⚠️")
    except Exception:
        st.error(f"🔴 API Offline — start backend with:\nuvicorn backend.app.main:app --port 8000", icon="🚨")

    st.markdown("---")

    nav_option = st.radio(
        "Navigation",
        [
            "🔍 Routine Conflict Checker",
            "📦 Product Catalog Search",
            "🧪 Ingredient Analysis",
            "📸 Packaging Label OCR",
            "📊 MLOps & Drift Telemetry"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🛒 Routine Basket")
    n = len(st.session_state.routine_products)
    st.write(f"**{n}** product{'s' if n != 1 else ''} staged")
    if n > 0:
        for idx, item in enumerate(st.session_state.routine_products):
            c1, c2 = st.columns([5, 1])
            c1.caption(f"{idx+1}. {item['brand']} — {item['name']}")
            if c2.button("✕", key=f"sb_del_{idx}"):
                st.session_state.routine_products.pop(idx)
                st.session_state.last_analysis = None
                st.rerun()
    if st.button("🗑️ Clear Basket", use_container_width=True):
        st.session_state.routine_products = []
        st.session_state.last_analysis = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# VIEW 1 — ROUTINE CONFLICT CHECKER
# ══════════════════════════════════════════════════════════════════════
if nav_option == "🔍 Routine Conflict Checker":
    st.markdown('<div class="header-title">Skincare Routine Conflict Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Add products, detect active conflicts, duplicate actives, and download an explained report.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.45])

    # ── Left: Product staging ──
    with col_left:
        st.subheader("1. Stage Products")
        add_mode = st.radio("Add via", ["Search Catalog", "Manual Paste"], horizontal=True)

        if add_mode == "Search Catalog":
            search_query = st.text_input("Product name or keyword", placeholder="e.g. CeraVe Moisturizer, Retinol")
            brand_filter = st.text_input("Brand filter (optional)", placeholder="e.g. The Ordinary")

            if search_query:
                with st.spinner("Searching..."):
                    params = {"q": search_query, "limit": 8}
                    if brand_filter.strip():
                        params["brand"] = brand_filter.strip()
                    resp = api_get("/products/search", params=params)

                if resp is None:
                    st.error("Search failed — backend may be offline.")
                elif not resp.get("results"):
                    st.info("No matching products found. Try a different term or use Manual Paste.")
                else:
                    st.write(f"Found **{resp['count']}** match(es):")
                    for prod in resp["results"]:
                        has_ing = prod.get("has_ingredients", False)
                        ing_snippet = (prod.get("ingredients_raw") or "")[:110]
                        with st.expander(
                            f"{'✅' if has_ing else '⚠️'} [{prod['match_type']} {prod['match_score']}%] "
                            f"{prod['brand']} — {prod['name']} ({prod['category']})"
                        ):
                            col_a, col_b = st.columns(2)
                            col_a.metric("Rating", f"⭐ {prod['rating']}")
                            col_b.metric("Price", f"${prod['price']:.2f}")
                            if has_ing:
                                st.caption(f"Ingredients preview: {ing_snippet}…")
                            else:
                                st.warning("No ingredient list on file for this product.")
                            if st.button("➕ Add to Routine", key=f"add_{prod['product_id']}"):
                                st.session_state.routine_products.append({
                                    "product_id": prod["product_id"],
                                    "name": prod["name"],
                                    "brand": prod["brand"],
                                    "category": prod["category"],
                                    "price": prod["price"],
                                    "rating": prod["rating"],
                                    "ingredients": prod.get("ingredients_raw"),
                                    "source": prod["source"]
                                })
                                st.success(f"Added {prod['name']} to routine!")
                                st.rerun()

        else:  # Manual Paste
            with st.form("manual_add_form"):
                m_brand = st.text_input("Brand", value="The Ordinary")
                m_name = st.text_input("Product Name", value="Niacinamide 10% + Zinc 1%")
                m_cat = st.selectbox("Category", ["Cleanser", "Treatment", "Moisturizer", "Eye cream", "Sun protect", "Face Mask"])
                m_ings = st.text_area(
                    "INCI Ingredient List (comma-separated)",
                    value="Aqua (Water), Niacinamide 10%, Pentylene Glycol, Zinc PCA 1%, Dimethyl Isosorbide, Tamarindus Indica Seed Gum, Xanthan gum, Isoceteth-20, Ethoxydiglycol, Phenoxyethanol, Chlorphenesin.",
                    height=120
                )
                if st.form_submit_button("Add Custom Product"):
                    if m_ings.strip():
                        st.session_state.routine_products.append({
                            "name": m_name,
                            "brand": m_brand,
                            "category": m_cat,
                            "price": 10.0,
                            "rating": 4.5,
                            "ingredients": m_ings,
                            "source": "Manual User Input"
                        })
                        st.success(f"Added {m_name}!")
                        st.rerun()
                    else:
                        st.error("Ingredient list cannot be empty.")

        # ── Staged products list ──
        st.markdown("---")
        st.subheader("Staged Products")
        if not st.session_state.routine_products:
            st.info("No products added yet. Search the catalog or paste ingredients above.")
        else:
            for idx, item in enumerate(st.session_state.routine_products):
                c1, c2 = st.columns([5, 1])
                with c1:
                    ing_ok = bool(item.get("ingredients"))
                    st.markdown(
                        f"**{idx+1}. [{item['category']}]** {item['brand']} — {item['name']}"
                        f" {'✅' if ing_ok else '⚠️ No ingredients'}"
                    )
                with c2:
                    if st.button("❌", key=f"del_{idx}"):
                        st.session_state.routine_products.pop(idx)
                        st.session_state.last_analysis = None
                        st.rerun()

        st.markdown("---")
        st.subheader("2. Routine Preferences")
        c_time = st.selectbox("Routine Time", ["both", "morning", "evening"])
        c_avoid = st.text_input("Ingredients to Avoid (comma-separated)", placeholder="e.g. fragrance, parabens, alcohol")
        c_concerns = st.multiselect(
            "Target Skin Concerns",
            ["dryness", "oiliness", "uneven texture", "appearance of blemishes"]
        )

        run_btn = st.button("🚀 Analyze Routine Compatibility", type="primary", use_container_width=True)

        if run_btn:
            if not st.session_state.routine_products:
                st.error("Add at least one product before running analysis.")
            else:
                avoid_tokens = [x.strip() for x in c_avoid.split(",") if x.strip()]
                payload = {
                    "products": st.session_state.routine_products,
                    "routine_time": c_time,
                    "avoid_ingredients": avoid_tokens,
                    "user_concerns": c_concerns
                }
                with st.spinner("Analyzing cross-product active interactions & classifying ingredient functions…"):
                    resp = api_post("/check-routine", payload, timeout=20)
                    if resp and "error" not in resp:
                        st.session_state.last_analysis = resp
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Analysis failed: {resp.get('error', 'Unknown error')}")

    # ── Right: Results ──
    with col_right:
        st.subheader("3. Analysis Results")

        if not st.session_state.last_analysis:
            st.info("👈 Add products and click **Analyze Routine Compatibility** to see results.")
        else:
            res = st.session_state.last_analysis
            status = res.get("primary_status", "NO_CONFIGURED_CONFLICT_FOUND")
            render_status_badge(status)

            # --- Novelty UI: Formulation-Aware Skin Barrier Load Meter ---
            b_load = res.get('barrier_load', {})
            if b_load:
                b_score = b_load.get('score', 0)
                b_level = b_load.get('level', 'LOW')
                b_label = b_load.get('level_label', '')
                b_summary = b_load.get('summary', '')
                card_class = 'barrier-card-green' if b_level == 'LOW' else ('barrier-card-amber' if b_level == 'MODERATE' else 'barrier-card-red')
                badge_emoji = '🛡️' if b_level == 'LOW' else ('⚠️' if b_level == 'MODERATE' else '🚨')
                
                st.markdown(f'''
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; font-size: 1.05rem;">{badge_emoji} Skin Barrier Load Index</span>
                        <span style="font-weight: 800; font-size: 1.25rem;">{b_score} / 100 ({b_label})</span>
                    </div>
                    <p style="margin: 0; font-size: 0.88rem; color: #334155;">{b_summary}</p>
                </div>
                ''', unsafe_allow_html=True)
                st.progress(min(1.0, b_score / 100.0))

            st.caption(f"Analyzed {res.get('total_products', 0)} formulations · {sum(p['ingredient_count'] for p in res.get('products_analyzed', []))} ingredients")

            tab_conflicts, tab_synergies, tab_barrier, tab_products, tab_sequence, tab_export = st.tabs([
                "⚠️ Conflicts & Actives", "✨ Clinical Synergies", "🛡️ Barrier Load Analysis", "🧴 Product Ingredients", "⏱️ Sequencing", "📄 Export Report"
            ])

            with tab_conflicts:
                # Warnings
                if res.get("warnings"):
                    st.markdown("#### 🔴 Potential Active Conflicts")
                    for w in res["warnings"]:
                        st.markdown(f"""
                        <div class="warning-card">
                            <strong>{w['title']}</strong><br/>
                            {w['description']}<br/>
                            <small style="color:#64748b;">Source: {w['source']}</small>
                        </div>
                        """, unsafe_allow_html=True)

                # Cautions
                if res.get("cautions"):
                    st.markdown("#### 🟡 Compatibility Cautions")
                    for c in res["cautions"]:
                        st.markdown(f"""
                        <div class="caution-card">
                            <strong>{c['title']}</strong><br/>
                            {c['description']}<br/>
                            <small style="color:#64748b;">Source: {c['source']}</small>
                        </div>
                        """, unsafe_allow_html=True)

                # Duplicate actives
                if res.get("duplicate_actives"):
                    st.markdown("#### 🟠 Duplicate Active Ingredients")
                    for d in res["duplicate_actives"]:
                        st.warning(
                            f"**{d['active_name']}** detected in: "
                            f"{', '.join(d['products_involved'])}. {d['explanation']}"
                        )

                # Avoid violations
                if res.get("avoid_violations"):
                    st.markdown("#### 🚫 Avoid-List Violations")
                    for av in res["avoid_violations"]:
                        st.error(
                            f"**{av.get('brand', '')} {av.get('product_name', '')}**: "
                            f"Matched '{av.get('avoid_query', '')}' → `{av.get('matched_ingredient', '')}`"
                        )

                # Insufficient info
                if res.get("insufficient_info"):
                    st.markdown("#### ℹ️ Incomplete Data")
                    for ii in res["insufficient_info"]:
                        st.markdown(f"""
                        <div class="info-card">
                            <strong>{ii.get('product_name', 'Product')}</strong>: {ii.get('issue', '')}<br/>
                            <small>{ii.get('recommendation', '')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                # Concern insights
                if res.get("concern_insights"):
                    st.markdown("#### 💡 Skin Concern Insights")
                    for ci in res["concern_insights"]:
                        icon = "✅" if ci["status"] == "MATCHED" else "ℹ️"
                        st.info(f"{icon} **{ci['concern']}**: {ci['message']}")

                if not any([res.get("warnings"), res.get("cautions"), res.get("duplicate_actives"),
                            res.get("avoid_violations")]):
                    st.success("No active conflicts, cautions, or avoid violations identified among configured rules.")

            with tab_synergies:
                st.markdown("#### ✨ Clinically Validated Active Synergies")
                st.caption("Detects beneficial multi-ingredient pairings supported by dermatology literature (e.g. photoprotection, barrier buffering).")
                synergies = res.get('active_synergies', [])
                if synergies:
                    for s in synergies:
                        st.markdown(f'''
                        <div class="synergy-card">
                            <div style="font-weight: 700; color: #065f46; font-size: 0.98rem;">🌟 {s['title']}</div>
                            <div style="margin: 4px 0 6px 0; font-size: 0.88rem; color: #1f2937;">
                                <strong>Active Ingredients:</strong> {', '.join(s['paired_ingredients'])}<br/>
                                {s['benefit']}
                            </div>
                            <small style="color: #047857;">Evidence: {s['source']}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.info("No clinical synergy pairs detected among the current products. Try adding pairings like Vitamin C + E + Ferulic or Retinol + Ceramides!")

            with tab_barrier:
                st.markdown("#### 🛡️ Formulation-Aware Barrier Load Breakdown")
                st.caption("Novelty Engine: Weights active chemical potency, leave-on vs rinse-off vehicle, ingredient positions, and soothing buffers.")
                b_info = res.get('barrier_load', {})
                if b_info:
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Irritation Score", f"{b_info.get('score', 0)} / 100")
                    bc2.metric("Active Stress Level", b_info.get('level', 'LOW'))
                    bc3.metric("Buffering Actives", b_info.get('buffering_ingredients_detected', 0))
                    bc4.metric("Barrier Relief Points", f"-{b_info.get('buffering_relief_points', 0)} pts")

                    st.markdown("**Contributing Active Loads by Vehicle & Potency:**")
                    contribs = b_info.get('contributing_actives', [])
                    if contribs:
                        df_contribs = pd.DataFrame(contribs)
                        df_contribs.columns = ["Product", "Category", "Active Chemical", "Vehicle Type", "Irritation Points"]
                        st.dataframe(df_contribs, use_container_width=True, hide_index=True)
                    else:
                        st.success("No harsh monitored actives detected in this routine.")

                    if b_info.get('clash_penalty_applied', 0) > 0 or b_info.get('duplicate_penalty_applied', 0) > 0:
                        st.warning(f"⚠️ Penalties applied: +{b_info.get('clash_penalty_applied', 0)} pts for formulation clashes, +{b_info.get('duplicate_penalty_applied', 0)} pts for duplicate actives.")

            with tab_products:
                st.markdown("#### Ingredient Function Analysis by Product")
                for prod in res.get("products_analyzed", []):
                    with st.expander(f"📦 {prod['brand']} — {prod['name']} ({prod['category']}) · {prod['ingredient_count']} ingredients"):
                        st.caption(f"Source: {prod.get('source', 'Local Dataset')} | Rating: ⭐ {prod.get('rating', 0)} | Price: ${prod.get('price', 0):.2f}")
                        ings_data = []
                        for ing in prod.get("ingredients", []):
                            conc_disp = ing.get("concentration_display", "Unknown / Not provided")
                            ings_data.append({
                                "Ingredient": ing["canonical_name"],
                                "Concentration": conc_disp,
                                "Function": f"{function_color(ing.get('functional_category', ''))} {ing.get('functional_category', 'Unknown')}",
                                "Confidence": f"{ing.get('classifier_confidence', 0) * 100:.1f}%"
                            })
                        if ings_data:
                            st.dataframe(pd.DataFrame(ings_data), use_container_width=True, hide_index=True)

            with tab_sequence:
                st.markdown("#### Recommended Application Sequence")
                seq = res.get("sequencing_recommendation", {})
                col_am, col_pm = st.columns(2)
                with col_am:
                    st.markdown("**☀️ Morning Order:**")
                    morning = seq.get("morning", [])
                    if morning:
                        for i, p in enumerate(morning, 1):
                            st.write(f"{i}. [{p['category']}] {p['brand']} — {p['name']}")
                    else:
                        st.caption("No morning products in routine.")
                with col_pm:
                    st.markdown("**🌙 Evening Order:**")
                    evening = seq.get("evening", [])
                    if evening:
                        for i, p in enumerate(evening, 1):
                            st.write(f"{i}. [{p['category']}] {p['brand']} — {p['name']}")
                    else:
                        st.caption("No evening products in routine.")

            with tab_export:
                st.markdown("#### Download Routine Assessment Report")
                report_lines = [
                    "# Skincare Routine Conflict & Compatibility Report",
                    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Primary Status: **{res.get('primary_status')}**",
                    "",
                    f"## Products Analyzed ({res.get('total_products', 0)})",
                ]
                for p in res.get("products_analyzed", []):
                    report_lines.append(f"- **[{p['category']}]** {p['brand']} — {p['name']} ({p['ingredient_count']} ingredients, source: {p.get('source', 'Local Dataset')})")

                # Include Novel Barrier Load & Synergies in exported report
                b_ld = res.get('barrier_load', {})
                if b_ld:
                    report_lines.extend([
                        "",
                        f"## Skin Barrier Load Score: {b_ld.get('score', 0)} / 100 ({b_ld.get('level_label', '')})",
                        f"> {b_ld.get('summary', '')}",
                        f"- Buffering soothing ingredients detected: {b_ld.get('buffering_ingredients_detected', 0)} (-{b_ld.get('buffering_relief_points', 0)} relief points)"
                    ])

                if res.get('active_synergies'):
                    report_lines.extend(["", "## Clinically Validated Active Synergies"])
                    for syn in res['active_synergies']:
                        report_lines.append(f"- **[SYNERGY] {syn['title']}**: {syn['benefit']} (Source: {syn['source']})")

                report_lines.extend(["", "## Active Conflicts & Cautions"])
                for w in res.get("warnings", []):
                    report_lines.append(f"- **[WARNING]** {w['title']}: {w['description']} (Source: {w['source']})")
                for c in res.get("cautions", []):
                    report_lines.append(f"- **[CAUTION]** {c['title']}: {c['description']} (Source: {c['source']})")

                if res.get("duplicate_actives"):
                    report_lines.append("")
                    report_lines.append("## Duplicate Actives")
                    for d in res["duplicate_actives"]:
                        report_lines.append(f"- **{d['active_name']}**: {d['explanation']}")

                report_lines.extend(["", "## Ingredient Functions by Product"])
                for p in res.get("products_analyzed", []):
                    report_lines.append(f"\n### {p['brand']} — {p['name']}")
                    for ing in p.get("ingredients", []):
                        report_lines.append(
                            f"- {ing['canonical_name']}: {ing.get('functional_category', 'Unknown')} "
                            f"(Concentration: {ing.get('concentration_display', 'Unknown')})"
                        )

                report_lines.extend(["", "## Disclaimer", res.get("disclaimer", "")])
                report_md = "\n".join(report_lines)

                st.download_button(
                    "📥 Download Report (.md)",
                    data=report_md,
                    file_name="skincare_routine_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                with st.expander("Preview Report"):
                    st.code(report_md[:3000], language="markdown")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic & Educational Disclaimer:</strong> This application is for machine learning demonstration purposes.
        Conflict assessments are based on configured rules reviewed against published dermatological literature.
        Absence of a configured conflict does not guarantee individual tolerability.
        Always patch-test new products and consult a board-certified dermatologist for persistent skin conditions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# VIEW 2 — PRODUCT CATALOG SEARCH
# ══════════════════════════════════════════════════════════════════════
elif nav_option == "📦 Product Catalog Search":
    st.markdown('<div class="header-title">Cosmetics Product Catalog</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Search by product name or brand. Three-level matching: Exact → Partial → Fuzzy Levenshtein.</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([3, 2, 1])
    with col_s1:
        s_query = st.text_input("Search product name or keyword", value="", placeholder="e.g. Hydrating Cleanser")
    with col_s2:
        s_brand = st.text_input("Brand filter (optional)", value="", placeholder="e.g. CeraVe")
    with col_s3:
        s_limit = st.number_input("Max results", min_value=1, max_value=50, value=15)

    if st.button("🔍 Search Catalog", use_container_width=True, type="primary"):
        if not s_query.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner("Searching product database…"):
                params = {"q": s_query, "limit": s_limit}
                if s_brand.strip():
                    params["brand"] = s_brand.strip()
                resp = api_get("/products/search", params=params)

            if resp is None:
                st.error("Backend is offline. Start the FastAPI server.")
            elif not resp.get("results"):
                st.info(f"No products found matching **{s_query}**. Try a shorter keyword or check spelling.")
            else:
                st.write(f"Found **{resp['count']}** product(s) for query: *{s_query}*")
                for r in resp["results"]:
                    has_ing = r.get("has_ingredients", False)
                    skin_suit = r.get("skin_suitability", {})
                    skin_tags = " · ".join(
                        key.capitalize() for key in ["combination", "dry", "normal", "oily", "sensitive"]
                        if skin_suit.get(key, 0) == 1
                    ) or "N/A"
                    with st.expander(
                        f"{'✅' if has_ing else '⚠️'} [{r['match_type']} · {r['match_score']}%] "
                        f"**{r['brand']}** — {r['name']} *({r['category']})*"
                    ):
                        cA, cB, cC = st.columns(3)
                        cA.metric("Price", f"${r['price']:.2f}")
                        cB.metric("Rating", f"⭐ {r['rating']}")
                        cC.metric("Match", f"{r['match_score']}% {r['match_type']}")

                        # Skin suitability
                        suited = [k.title() for k, v in skin_suit.items() if v == 1]
                        if suited:
                            st.caption(f"Suitable for: {', '.join(suited)}")

                        # Ingredient list
                        st.markdown("**Full Ingredient List:**")
                        if has_ing:
                            st.text_area(
                                "INCI List",
                                value=r.get("ingredients_raw", ""),
                                height=100,
                                key=f"ing_view_{r['product_id']}",
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        else:
                            st.info("No ingredient list available for this product. Use Manual Paste in the Routine Checker.")

                        st.caption(f"Data Source: {r.get('source', 'Local Dataset')} | ID: {r['product_id']}")

                        if st.button(f"➕ Add to Routine Basket", key=f"cat_add_{r['product_id']}"):
                            st.session_state.routine_products.append({
                                "product_id": r["product_id"],
                                "name": r["name"],
                                "brand": r["brand"],
                                "category": r["category"],
                                "price": r["price"],
                                "rating": r["rating"],
                                "ingredients": r.get("ingredients_raw"),
                                "source": r.get("source", "Local Dataset")
                            })
                            st.success(f"Added **{r['name']}** to your routine basket!")


# ══════════════════════════════════════════════════════════════════════
# VIEW 3 — INGREDIENT ANALYSIS (STANDALONE)
# ══════════════════════════════════════════════════════════════════════
elif nav_option == "🧪 Ingredient Analysis":
    st.markdown('<div class="header-title">Ingredient Analysis & Function Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Paste a raw ingredient list to parse, normalize synonyms, classify functions, and extract concentrations. Or classify a single ingredient.</div>', unsafe_allow_html=True)

    tab_bulk, tab_single = st.tabs(["📋 Bulk Ingredient List", "🔬 Single Ingredient Lookup"])

    # ── Tab A: Bulk parse ──
    with tab_bulk:
        st.markdown("#### Paste Ingredient List (INCI format, comma-separated)")
        default_ings = (
            "Aqua (Water), Niacinamide 10%, Pentylene Glycol, Zinc PCA 1%, "
            "Glycerin, Retinol 0.3%, Salicylic Acid 2%, Sodium Hyaluronate, "
            "Tocopheryl Acetate, Phenoxyethanol, Centella Asiatica Extract"
        )
        raw_text = st.text_area(
            "Ingredient list",
            value=default_ings,
            height=130,
            placeholder="Paste full INCI ingredient list here...",
            label_visibility="collapsed"
        )

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            parse_btn = st.button("🔬 Parse & Classify", type="primary", use_container_width=True)
        with col_b2:
            if st.button("➕ Add Parsed Product to Routine", use_container_width=True):
                if raw_text.strip():
                    brand_i = st.session_state.get("bulk_brand", "Manual Input")
                    name_i = st.session_state.get("bulk_name", "Pasted Product")
                    st.session_state.routine_products.append({
                        "name": name_i, "brand": brand_i,
                        "category": "Treatment", "price": 0.0, "rating": 4.0,
                        "ingredients": raw_text, "source": "Manual Text Input"
                    })
                    st.success("Product added to routine basket!")

        if parse_btn:
            if not raw_text.strip():
                st.warning("Please paste an ingredient list first.")
            else:
                with st.spinner("Parsing & classifying ingredients…"):
                    res = api_post("/ingredients/parse", {"ingredients_text": raw_text}, timeout=20)

                if res is None or "error" in res:
                    st.error(f"Parse failed: {res.get('error') if res else 'Backend offline'}")
                else:
                    st.session_state.parsed_ingredient_results = res
                    st.success(f"Parsed **{res['ingredient_count']}** ingredients.")

        # Display results
        if st.session_state.parsed_ingredient_results:
            res = st.session_state.parsed_ingredient_results
            st.markdown(f"### Results — {res['ingredient_count']} Ingredients Parsed")

            # Summary metrics
            ings = res.get("parsed_ingredients", [])
            has_conc = sum(1 for i in ings if i.get("has_explicit_concentration"))
            unknown_ct = sum(1 for i in ings if "Unknown" in (i.get("functional_category") or ""))
            functions = [i.get("functional_category", "") for i in ings if "Unknown" not in (i.get("functional_category") or "")]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Ingredients", res["ingredient_count"])
            mc2.metric("With Explicit Concentration", has_conc)
            mc3.metric("Classified Functions", len(functions))
            mc4.metric("Unknown / Low-Confidence", unknown_ct)

            st.markdown("---")

            # Function legend
            st.markdown("**Function Legend:**")
            legend_cols = st.columns(4)
            function_list = ["Humectant", "Emollient", "Occlusive", "Exfoliating agent",
                             "Antioxidant", "Surfactant", "Soothing / Anti-inflammatory", "Sunscreen active"]
            for i, f in enumerate(function_list):
                legend_cols[i % 4].caption(f"{function_color(f)} {f}")

            st.markdown("---")

            # Full table
            rows = []
            for ing in ings:
                conc = ing.get("concentration")
                conc_src = ing.get("concentration_source", "Not provided")
                rows.append({
                    "Original Token": ing.get("raw_token", ""),
                    "Canonical Name": ing.get("canonical_name", ""),
                    "Concentration": f"{conc}%" if conc is not None else "—",
                    "Concentration Source": conc_src,
                    "Function": f"{function_color(ing.get('functional_category', ''))} {ing.get('functional_category', 'Unknown')}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Synonym normalization highlights
            normalized = [(i["raw_token"], i["canonical_name"]) for i in ings
                          if i["raw_token"].lower().strip() != i["canonical_name"].lower().strip()]
            if normalized:
                st.markdown("#### 🔄 Synonym Normalizations Applied")
                for raw, canon in normalized:
                    st.caption(f"`{raw}` → **{canon}**")

    # ── Tab B: Single ingredient ──
    with tab_single:
        st.markdown("#### Classify a Single Ingredient")

        col_i1, col_i2 = st.columns([1.2, 1])
        with col_i1:
            ing_name = st.text_input("Ingredient Name", value="Glycerin", placeholder="e.g. Niacinamide, Salicylic Acid, Squalane")

            # Quick example buttons
            st.markdown("**Quick examples:**")
            q_cols = st.columns(4)
            examples = ["Glycerin", "Salicylic Acid", "Squalane", "Niacinamide",
                        "Avobenzone", "Dimethicone", "Centella Asiatica", "Ascorbic Acid"]
            for i, ex in enumerate(examples):
                if q_cols[i % 4].button(ex, key=f"ex_{i}"):
                    ing_name = ex

            classify_btn = st.button("🔬 Classify Ingredient", type="primary", use_container_width=True)

        with col_i2:
            if classify_btn or ing_name:
                with st.spinner(f"Classifying {ing_name}…"):
                    res = api_post("/ingredients/classify", {"ingredient_name": ing_name})

                if res and "error" not in res:
                    func = res.get("predicted_function", "Unknown")
                    conf = res.get("confidence", 0.0)

                    if "Unknown" in func:
                        st.warning(f"**{ing_name}** → `Unknown / Insufficient evidence`")
                        st.caption(f"Confidence ({conf*100:.1f}%) is below threshold. The ingredient may be novel or absent from the training taxonomy.")
                    else:
                        st.success(f"**{func}** {function_color(func)}")
                        st.metric("Model Confidence", f"{conf*100:.1f}%")

                    if res.get("probabilities"):
                        st.markdown("#### Class Probability Distribution")
                        prob_df = pd.DataFrame(
                            list(res["probabilities"].items()),
                            columns=["Function", "Probability"]
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Function"))

                    st.markdown("---")
                    st.caption(
                        "**Limitation:** The classifier is trained on a curated 77-record INCI taxonomy. "
                        "Novel ingredients or trade names may return 'Unknown / Insufficient evidence'. "
                        "Results should be verified against the INCI database for clinical use."
                    )
                else:
                    st.error(f"Classification failed: {res.get('error') if res else 'Backend offline'}")


# ══════════════════════════════════════════════════════════════════════
# VIEW 4 — PACKAGING LABEL OCR
# ══════════════════════════════════════════════════════════════════════
elif nav_option == "📸 Packaging Label OCR":
    st.markdown('<div class="header-title">Packaging Label Image OCR</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Upload a product packaging photograph. The backend uses Tesseract OCR (Pillow preprocessing). Review and edit extracted text before analysis.</div>', unsafe_allow_html=True)

    col_o1, col_o2 = st.columns([1, 1.2])

    with col_o1:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a product packaging image",
            type=["jpg", "jpeg", "png"],
            help="Best results with clear, high-contrast text. Max 10MB."
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            st.caption(f"Size: {len(uploaded_file.getvalue()) / 1024:.1f} KB | Type: {uploaded_file.type}")

            if st.button("🔍 Extract Text via OCR", type="primary", use_container_width=True):
                with st.spinner("Sending to Tesseract OCR engine…"):
                    try:
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        resp = requests.post(f"{API_URL}/ocr/extract", files=files, timeout=20)
                        if resp.status_code == 200:
                            ocr_data = resp.json()
                            st.session_state.ocr_last_result = ocr_data
                            if ocr_data["status"] == "SUCCESS":
                                st.session_state.ocr_text_buffer = ocr_data["extracted_text"]
                                st.success("✅ Text extracted. Review and correct on the right.")
                            elif ocr_data["status"] == "OCR_ENGINE_UNAVAILABLE":
                                st.warning("⚠️ Tesseract OCR binary not found on this host.")
                                st.info(ocr_data.get("notes", ""))
                            else:
                                st.error(f"OCR Error: {ocr_data.get('error', 'Unknown')}")
                                st.info(ocr_data.get("notes", ""))
                        else:
                            st.error(f"Backend returned error {resp.status_code}: {resp.text[:200]}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")

    with col_o2:
        st.subheader("2. Review & Edit Extracted Text")
        st.caption("Tesseract OCR may misread certain characters (e.g. rn→m). Review carefully before analysis.")

        edited_text = st.text_area(
            "Extracted ingredient text (edit as needed):",
            value=st.session_state.ocr_text_buffer,
            height=220,
            placeholder="Extracted ingredient text will appear here. You may also paste any ingredient list manually.",
        )

        # Quick parse preview
        if edited_text.strip() and st.button("🔬 Preview Parsed Ingredients", use_container_width=True):
            with st.spinner("Parsing…"):
                preview_res = api_post("/ingredients/parse", {"ingredients_text": edited_text})
            if preview_res and "error" not in preview_res:
                st.success(f"Parsed {preview_res['ingredient_count']} ingredients from OCR text.")
                rows = [
                    {"Canonical Name": i["canonical_name"],
                     "Concentration": f"{i['concentration']}%" if i.get("concentration") else "—",
                     "Function": i.get("functional_category", "Unknown")}
                    for i in preview_res.get("parsed_ingredients", [])
                ]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("3. Add to Routine")

        ocr_brand = st.text_input("Brand (optional)", value="OCR Scanned")
        ocr_name = st.text_input("Product Name (optional)", value="Packaging Scan")
        ocr_cat = st.selectbox("Category", ["Treatment", "Cleanser", "Moisturizer", "Eye cream", "Sun protect", "Face Mask"])

        if st.button("➕ Add to Routine Basket", type="primary", use_container_width=True):
            if edited_text.strip():
                st.session_state.routine_products.append({
                    "name": ocr_name,
                    "brand": ocr_brand,
                    "category": ocr_cat,
                    "price": 0.0,
                    "rating": 4.0,
                    "ingredients": edited_text.strip(),
                    "source": "Image OCR Ingestion (User Verified)"
                })
                st.success(f"Added **{ocr_name}** to routine basket! Go to the Routine Conflict Checker to analyze.")
            else:
                st.warning("Ingredient text cannot be empty.")


# ══════════════════════════════════════════════════════════════════════
# VIEW 5 — MLOPS & DRIFT TELEMETRY
# ══════════════════════════════════════════════════════════════════════
elif nav_option == "📊 MLOps & Drift Telemetry":
    st.markdown('<div class="header-title">MLOps Architecture & Drift Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Monitor model versions, training lineage, promotion history, drift indicators, and operational latency.</div>', unsafe_allow_html=True)

    # Pull model info and metrics
    m_info = api_get("/model-info") or {}
    metrics_data = api_get("/metrics") or {}
    drift_data = api_get("/drift") or {}
    ops_data = metrics_data.get("operational", {})

    # ── Model Versions ──
    st.subheader("🏷️ Active Model Versions")
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Product Classifier Version", f"v{m_info.get('current_production_version', '?')}")
    cm2.metric("Model Architecture", m_info.get("architecture", "RandomForest"))
    cm3.metric("Status", m_info.get("status", "UNKNOWN"))
    cm4.metric("Promoted At", m_info.get("promoted_at", "—")[:10] if m_info.get("promoted_at") else "—")

    prod_metrics = m_info.get("metrics", {})
    if prod_metrics:
        cm5, cm6, cm7 = st.columns(3)
        cm5.metric("Val Accuracy", f"{prod_metrics.get('accuracy', 0)*100:.1f}%")
        cm6.metric("Macro F1", f"{prod_metrics.get('f1_macro', 0):.4f}")
        cm7.metric("Weighted F1", f"{prod_metrics.get('f1_weighted', 0):.4f}")

    ing_cls_info = m_info.get("ingredient_classifier", {})
    if ing_cls_info:
        st.markdown("---")
        st.subheader("🧪 Ingredient Function Classifier")
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Type", ing_cls_info.get("model_type", "—")[:20])
        ic2.metric("Classes", len(ing_cls_info.get("supported_classes", [])))
        ic3.metric("Val Accuracy", f"{ing_cls_info.get('validation_accuracy', 0)*100:.1f}%")
        ic4.metric("Macro F1", f"{ing_cls_info.get('validation_macro_f1', 0):.4f}")

        if ing_cls_info.get("supported_classes"):
            with st.expander("Supported Functional Classes"):
                for cls in ing_cls_info["supported_classes"]:
                    st.write(f"  {function_color(cls)} **{cls}**")

        if ing_cls_info.get("limitations"):
            st.info(f"⚠️ **Limitation Note:** {ing_cls_info['limitations']}")

    # ── Drift Dashboard ──
    st.markdown("---")
    st.subheader("📉 Data & Prediction Drift")

    if not drift_data:
        st.warning("Drift data unavailable — ensure backend is running and at least one inference has been made.")
    else:
        severity = drift_data.get("drift_severity", "UNKNOWN")
        sev_color = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}.get(severity, "⚪")
        st.markdown(f"**Overall Drift Severity:** {sev_color} {severity}")

        vocab = drift_data.get("vocabulary_drift", {})
        pred = drift_data.get("prediction_drift", {})

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Tokens Observed", vocab.get("total_tokens_observed", 0))
        d2.metric("Unseen Rate (Cumulative)", f"{vocab.get('cumulative_unseen_rate', 0)*100:.1f}%")
        d3.metric("Unseen Rate (Recent Window)", f"{vocab.get('recent_unseen_rate', 0)*100:.1f}%")
        d4.metric("Unique New Terms", vocab.get("unique_unseen_terms", 0))

        dp1, dp2, dp3 = st.columns(3)
        dp1.metric("Total Classifications", pred.get("total_classifications", 0))
        dp2.metric("Unknown Predictions", pred.get("unknown_classifications", 0))
        dp3.metric("Unknown Rate", f"{pred.get('unknown_prediction_rate', 0)*100:.1f}%")

        if vocab.get("sample_unseen_terms"):
            with st.expander("🔍 Sample Unseen Ingredients (last 10 novel terms)"):
                for term in vocab["sample_unseen_terms"]:
                    st.caption(f"• {term}")

        # Class distribution shift
        class_shift = pred.get("class_distribution_shift", {})
        if class_shift.get("status") == "COMPUTED" and class_shift.get("details"):
            st.markdown("**Prediction Distribution vs. Training Reference:**")
            shift_rows = []
            for cls, vals in class_shift["details"].items():
                shift_rows.append({
                    "Class": cls,
                    "Training %": f"{vals['reference']*100:.1f}%",
                    "Live %": f"{vals['live']*100:.1f}%",
                    "Delta": f"{vals['delta']:+.4f}"
                })
            st.dataframe(pd.DataFrame(shift_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Class distribution shift requires at least one /ingredients/classify inference.")

        # Limitation note
        with st.expander("📌 Interpretation Notes"):
            st.info(drift_data.get("statistical_limitation_note", ""))
            st.caption(f"Reference vocabulary size: {drift_data.get('reference_vocab_size', 0)} terms")
            st.caption(f"Report generated at: {drift_data.get('generated_at', '—')}")

    # ── Latency Summary ──
    latency_data = drift_data.get("latency_summary", {})
    if latency_data:
        st.markdown("---")
        st.subheader("⏱️ API Latency Summary")
        lat_rows = []
        for ep, vals in latency_data.items():
            lat_rows.append({
                "Endpoint": ep,
                "P50 (ms)": vals.get("p50_ms", "—"),
                "P95 (ms)": vals.get("p95_ms", "—"),
                "Request Count": vals.get("count", 0)
            })
        if lat_rows:
            st.dataframe(pd.DataFrame(lat_rows), use_container_width=True, hide_index=True)

    # ── Operational Telemetry ──
    st.markdown("---")
    st.subheader("⚙️ Operational Telemetry (Raw)")
    if ops_data:
        with st.expander("View full operational metrics JSON"):
            st.json(ops_data)
    else:
        st.caption("Operational telemetry not available.")

    # ── MLOps Pipeline diagram ──
    st.markdown("---")
    st.subheader("🔄 MLOps Pipeline")
    st.markdown("""
    ```
    Dataset Download (SHA-256 verified)
      ↓
    Schema & Bounds Validation
      ↓
    Preprocessing (INCI synonym normalization · concentration extraction · stratified split)
      ↓
    Feature Engineering (TF-IDF word N-grams + tabular FeatureUnion)
      ↓
    Multi-Model Training (LR baseline · Random Forest → MLflow tracked runs)
      ↓
    Gated Promotion  [Accuracy ≥ 40% AND Macro-F1 ≥ 40% AND inference smoke test]
      ↓
    Versioned Model Registry  (models/model_registry.json · rollback chain)
      ↓
    FastAPI Production Serving  (/predict · /check-routine · /ingredients/classify)
      ↓
    Drift Monitoring  (unseen rate · unknown prediction rate · latency p50/p95)
      ↓
    Rollback on Demand  (python scripts/retrain.py --rollback)
    ```
    """)
