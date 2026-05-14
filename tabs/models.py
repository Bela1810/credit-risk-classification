# ─────────────────────────────────────────────
# tabs/models.py — Tab 3: Model Results
# ─────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config import COLORS, FRIENDLY_NAMES, MODEL_COLORS, PLOTLY_TEMPLATE


def _resolve_classifier_and_feature_names(model, fallback_feat_names: list):
    """
    Return (classifier, feature_names) for either a bare estimator or a
    sklearn Pipeline that ends in a classifier (optionally preceded by a
    preprocessor / ColumnTransformer that expands feature names).
    """
    if hasattr(model, "named_steps"):
        clf = model[-1]
        try:
            names = list(model[:-1].get_feature_names_out())
        except Exception:
            names = fallback_feat_names
    else:
        clf = model
        names = fallback_feat_names
    return clf, names


def _prettify_feature_name(name: str) -> str:
    """Turn ColumnTransformer names like
    'categoricas_nominales__cartera_consumo_con_libranza' into
    'Tipo de Cartera — consumo_con_libranza'."""
    bare = name.split("__", 1)[1] if "__" in name else name
    for col, friendly in FRIENDLY_NAMES.items():
        if bare == col:
            return friendly
        prefix = f"{col}_"
        if bare.startswith(prefix):
            return f"{friendly} — {bare[len(prefix):]}"
    return bare


def render(
    models: dict,
    results: dict,
    model_names: list,
    feat_names: list,
    y_test,
    threshold: float,
) -> None:

    # ── Metric cards ─────────────────────────────
    st.markdown('<div class="section-title">Métricas de Clasificación</div>',
                unsafe_allow_html=True)
    st.caption(
        "**Recall** mide qué porcentaje de los defaults reales detecta el modelo "
        "(la métrica priorizada en este proyecto). El umbral de decisión se ajusta "
        "en la barra lateral."
    )
    col1, col2 = st.columns(2)
    for col, mname in zip([col1, col2], model_names):
        with col:
            badge = "badge-gbc" if "Boosting" in mname else "badge-lgbm"
            st.markdown(f'<span class="model-badge {badge}">{mname}</span>',
                        unsafe_allow_html=True)
            m = results[mname]["metrics"]
            a, b, c, d, e = st.columns(5)
            a.metric("Exactitud",   f'{m["Exactitud"]:.2%}')
            b.metric("Recall",      f'{m["Recall"]:.4f}',
                     help="Proporción de defaults reales correctamente detectados (TP / (TP + FN)).")
            c.metric("F1",          f'{m["F1"]:.4f}')
            d.metric("ROC-AUC",     f'{m["ROC-AUC"]:.4f}')
            e.metric("Prec. Prom.", f'{m["Prec. Prom."]:.4f}')

    # ── Side-by-side bar ─────────────────────────
    st.markdown('<div class="section-title">Comparación Lado a Lado de Métricas</div>',
                unsafe_allow_html=True)
    cmp_rows = [
        {"Modelo": m, "Métrica": k, "Valor": v}
        for m in model_names
        for k, v in results[m]["metrics"].items()
    ]
    fig_cmp = px.bar(
        pd.DataFrame(cmp_rows), x="Métrica", y="Valor",
        color="Modelo", barmode="group",
        color_discrete_map=MODEL_COLORS,
        text_auto=".4f", template=PLOTLY_TEMPLATE,
        title="Gradient Boosting vs LightGBM",
    )
    fig_cmp.update_layout(height=380, margin=dict(t=50, b=40))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── ROC Curves ───────────────────────────────
    st.markdown('<div class="section-title">Curvas ROC</div>',
                unsafe_allow_html=True)
    fig_roc = go.Figure()
    for mname, color in MODEL_COLORS.items():
        fpr, tpr, _ = roc_curve(y_test, results[mname]["y_prob"])
        auc_val      = results[mname]["metrics"]["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{mname}  (AUC = {auc_val:.4f})",
            line=dict(color=color, width=2.5),
        ))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="gray", dash="dash", width=1.5))
    fig_roc.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos (Recall)",
        title=f"Curvas ROC  (umbral = {threshold:.2f})",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=70, l=50, r=20),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Precision-Recall Curves ──────────────────
    st.markdown('<div class="section-title">Curvas Precisión-Recall</div>',
                unsafe_allow_html=True)
    fig_pr = go.Figure()
    for mname, color in MODEL_COLORS.items():
        prec, rec, _ = precision_recall_curve(y_test, results[mname]["y_prob"])
        ap_val        = results[mname]["metrics"]["Prec. Prom."]
        recall_val    = results[mname]["metrics"]["Recall"]
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{mname}  (AP = {ap_val:.4f}, Recall@umbral = {recall_val:.3f})",
            line=dict(color=color, width=2.5),
        ))
    fig_pr.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                     annotation_text=f"Línea base ({y_test.mean():.2f})")
    fig_pr.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        xaxis_title="Recall", yaxis_title="Precisión",
        title="Curvas Precisión-Recall",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=70, l=50, r=20),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion Matrices ───────────────────────
    st.markdown('<div class="section-title">Matrices de Confusión</div>',
                unsafe_allow_html=True)
    col_cm1, col_cm2 = st.columns(2)
    labels = ["Sin Default", "Default"]
    for col, mname in zip([col_cm1, col_cm2], model_names):
        with col:
            cm = confusion_matrix(y_test, results[mname]["y_pred"])
            fig_cm = px.imshow(
                cm, text_auto=True, x=labels, y=labels,
                color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE, title=mname,
                labels={"x": "Predicho", "y": "Real"},
            )
            fig_cm.update_layout(height=320,
                                 margin=dict(t=50, b=40, l=60, r=20))
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Probability Distributions ────────────────
    st.markdown('<div class="section-title">Distribución de Probabilidades Predichas</div>',
                unsafe_allow_html=True)
    prob_sel = st.radio("Modelo", model_names, horizontal=True, key="prob_radio")
    prob_df  = pd.DataFrame({
        "Probabilidad": results[prob_sel]["y_prob"],
        "Clase Real":   pd.Series(y_test.values).map({0: "Sin Default", 1: "Default"}),
    })
    fig_prob = px.histogram(
        prob_df, x="Probabilidad", color="Clase Real",
        color_discrete_map={"Sin Default": COLORS[0], "Default": COLORS[4]},
        nbins=60, barmode="overlay", opacity=0.75,
        template=PLOTLY_TEMPLATE,
        title=f"{prob_sel} — Probabilidad Predicha por Clase Real",
    )
    fig_prob.add_vline(x=threshold, line_dash="dash", line_color="white",
                       annotation_text=f"Umbral ({threshold})")
    fig_prob.update_layout(height=360, margin=dict(t=50, b=40))
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Feature Importance ───────────────────────
    st.markdown('<div class="section-title">Importancia de Variables</div>',
                unsafe_allow_html=True)
    top_n = st.slider("Top N variables", 5, 25, 15, 1)

    col_fi1, col_fi2 = st.columns(2)
    for col, mname in zip([col_fi1, col_fi2], model_names):
        with col:
            clf, names = _resolve_classifier_and_feature_names(
                models[mname], feat_names,
            )
            if not hasattr(clf, "feature_importances_"):
                st.warning(f"{mname} no expone `feature_importances_`.")
                continue
            importances = clf.feature_importances_
            if len(importances) != len(names):
                st.warning(
                    f"{mname}: la cantidad de importancias ({len(importances)}) "
                    f"no coincide con la cantidad de nombres ({len(names)})."
                )
                continue
            fi_df = (
                pd.DataFrame({"Variable": names, "Importancia": importances})
                .assign(Variable=lambda d: d["Variable"].map(_prettify_feature_name))
                .nlargest(top_n, "Importancia")
                .sort_values("Importancia")
            )
            scale = "Purples" if "Boosting" in mname else "Greens"
            fig_fi = px.bar(
                fi_df, x="Importancia", y="Variable", orientation="h",
                color="Importancia", color_continuous_scale=scale,
                template=PLOTLY_TEMPLATE,
                title=f"{mname} — Top {top_n} Variables",
            )
            fig_fi.update_layout(height=50 * top_n + 80, showlegend=False,
                                 margin=dict(t=50, b=40, l=20, r=20))
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Full Classification Report ───────────────
    with st.expander("📄 Reporte Completo de Clasificación"):
        for mname in model_names:
            st.markdown(f"**{mname}**")
            report = classification_report(
                y_test, results[mname]["y_pred"],
                target_names=["Sin Default", "Default"],
                output_dict=True,
            )
            st.dataframe(
                pd.DataFrame(report).T.style
                .format("{:.4f}")
                .background_gradient(
                    cmap="Blues",
                    subset=["f1-score", "precision", "recall"],
                ),
                use_container_width=True,
            )
            st.markdown("---")
