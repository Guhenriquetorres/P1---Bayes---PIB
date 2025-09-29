# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ====================== CONFIG BÁSICO ======================
st.set_page_config(page_title="Exploração & Previsões — PIB Setor (Interativo)", layout="wide")
BASE_DIR = Path("/content/drive/MyDrive/PIB_Forecast")
PLOTLY_TEMPLATE = "plotly_white"

# ====================== CSS ======================
st.markdown("""
<style>
html, body, [class*="css"]{font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.section{border-radius:18px;border:1px solid rgba(0,0,0,.06);padding:18px;background:#fff;box-shadow:0 8px 24px rgba(0,0,0,.04)}
.metric-card{border-radius:16px;padding:16px 18px;border:1px solid rgba(0,0,0,.06);background:linear-gradient(180deg,rgba(0,0,0,.03),rgba(0,0,0,.015));box-shadow:0 4px 12px rgba(0,0,0,.05)}
.small{color:#666;font-size:.9rem}
h1,h2,h3{letter-spacing:.2px}
hr{margin:.6rem 0 1rem 0;border-color:rgba(0,0,0,.08)}
</style>
""", unsafe_allow_html=True)

# ====================== HELPERS ======================
@st.cache_data(show_spinner=False)
def _read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if not p.exists(): return None
        try:
            df = pd.read_csv(p, dtype={"ano_pib": "Int64"})
        except Exception:
            df = pd.read_csv(p)
        if "ano_pib" in df.columns:
            df["ano_pib"] = pd.to_numeric(df["ano_pib"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        st.warning(f"Falha ao ler **{p.name}**: {e}")
        return None

@st.cache_data(show_spinner=False)
def _read_cm(p: Path) -> pd.DataFrame | None:
    try:
        if not p.exists(): return None
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str); df.columns = df.columns.astype(str)
        if list(df.index) != list(df.columns):
            labs = list(dict.fromkeys(list(df.index)+list(df.columns)))
            df = df.reindex(index=labs, columns=labs, fill_value=0)
        return df
    except Exception as e:
        st.warning(f"Falha ao ler **{p.name}**: {e}")
        return None

@st.cache_data(show_spinner=False)
def _read_txt(p: Path) -> str | None:
    try:
        if p.exists(): return p.read_text(encoding="utf-8")
    except Exception as e:
        st.warning(f"Falha ao ler **{p.name}**: {e}")
    return None

def _download(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", use_container_width=True)

def _normalize_cm(cm: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "Contagem": return cm
    A = cm.values.astype(float)
    if how == "% por linha (verdadeiro)":
        row = A.sum(axis=1, keepdims=True); row[row==0]=1; A = 100*A/row
    elif how == "% por coluna (predito)":
        col = A.sum(axis=0, keepdims=True); col[col==0]=1; A = 100*A/col
    elif how == "% do total":
        s = A.sum() or 1; A = 100*A/s
    return pd.DataFrame(A, index=cm.index, columns=cm.columns).round(2)

def _anos_unicos_as_str(df: pd.DataFrame) -> list[str]:
    if "ano_pib" not in df.columns: return []
    anos = pd.to_numeric(df["ano_pib"], errors="coerce").dropna().astype(int).unique().tolist()
    return [str(a) for a in sorted(anos)]

def _aplicar_filtro_ano(df: pd.DataFrame, ano_str: str) -> pd.DataFrame:
    if ano_str == "(todos)" or "ano_pib" not in df.columns: return df
    ano_int = int(ano_str)
    return df[pd.to_numeric(df["ano_pib"], errors="coerce").astype("Int64") == ano_int]

def _count_df(series: pd.Series, label_name="setor") -> pd.DataFrame:
    """Conserta duplicidade de nomes após value_counts()."""
    s = series.dropna().astype(str).value_counts()
    return s.rename_axis(label_name).reset_index(name="count")

# ====================== ARQUIVOS ======================
RES_BAYES = BASE_DIR / "results_classify_test_BAYES.csv"
SUM_BAYES = BASE_DIR / "metrics_classify_summary_BAYES.csv"
CM_BAYES  = BASE_DIR / "metrics_classify_confusion_matrix_BAYES.csv"
TXT_BAYES = BASE_DIR / "classification_report_BAYES.txt"
RES_FREQ  = BASE_DIR / "results_classify_test.csv"
SUM_FREQ  = BASE_DIR / "metrics_classify_summary.csv"
CM_FREQ   = BASE_DIR / "metrics_classify_confusion_matrix_freq.csv"

df_res_bayes = _read_csv(RES_BAYES)
df_sum_bayes = _read_csv(SUM_BAYES)
df_cm_bayes  = _read_cm(CM_BAYES)
txt_bayes    = _read_txt(TXT_BAYES)
df_res_freq  = _read_csv(RES_FREQ)
df_sum_freq  = _read_csv(SUM_FREQ)
df_cm_freq   = _read_cm(CM_FREQ)

# ====================== HEADER ======================
st.markdown("<h1>🔎 Exploração & 🎯 Previsões — PIB por Setor (Interativo)</h1>", unsafe_allow_html=True)
with st.expander("Arquivos detectados", expanded=False):
    c1, c2 = st.columns(2)
    c1.write(f"**Bayes**: "
             f"{'✔️' if df_res_bayes is not None else '❌'} results | "
             f"{'✔️' if df_sum_bayes is not None else '❌'} summary | "
             f"{'✔️' if df_cm_bayes is not None else '❌'} confusion")
    c2.write(f"**Frequente**: "
             f"{'✔️' if df_res_freq is not None else '❌'} results | "
             f"{'✔️' if df_sum_freq is not None else '❌'} summary | "
             f"{'✔️' if df_cm_freq is not None else '❌'} confusion")

# ====================== SIDEBAR ======================
st.sidebar.header("Fonte dos resultados")
opts = []
if df_res_bayes is not None: opts.append("Bayes (ADVI)")
if df_res_freq  is not None: opts.append("Frequente (Logistic)")
if not opts:
    st.error("Nenhum resultado encontrado em /content/drive/MyDrive/PIB_Forecast.")
    st.stop()
src_choice = st.sidebar.radio("Modelo:", opts, index=0)

df_base = (df_res_bayes.copy() if src_choice.startswith("Bayes") else df_res_freq.copy())
st.sidebar.subheader("Filtros globais")
anos_str = _anos_unicos_as_str(df_base)
ano_sel = st.sidebar.selectbox("Ano", ["(todos)"]+anos_str, index=0)
q_muni  = st.sidebar.text_input("Buscar município (nome ou código)")
setores = sorted(df_base["setor"].dropna().unique().tolist()) if "setor" in df_base.columns else []
set_true = st.sidebar.selectbox("Setor verdadeiro", ["(todos)"]+setores, index=0)

df_filtered = _aplicar_filtro_ano(df_base, ano_sel)
if q_muni.strip():
    q = q_muni.lower().strip()
    df_filtered = df_filtered[
        (df_filtered["nome_municipio"].astype(str).str.lower().str.contains(q)) |
        (df_filtered["cod_mun"].astype(str).str.contains(q))
    ]
if set_true != "(todos)" and "setor" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["setor"] == set_true]

# ====================== TABS ======================
tab_exp, tab_prev = st.tabs(["✨ Análises Exploratórias", "🎯 Previsões"])

# ====================== EXPLORATÓRIA ======================
with tab_exp:
    if df_filtered.empty:
        st.info("Sem dados para os filtros atuais."); st.stop()

    left, right = st.columns(2, gap="large")

    # -------- Esquerda: visão geral + evolução --------
    with left:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Visão geral")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="small">Observações</div><h3>{len(df_filtered):,}</h3></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="small">Municípios</div><h3>{df_filtered["cod_mun"].nunique():,}</h3></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="small">Anos</div><h3>{df_filtered["ano_pib"].nunique()}</h3></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="small">Setores</div><h3>{df_filtered["setor"].nunique()}</h3></div>', unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        order_by = st.selectbox("Ordenar barras por:", ["Contagem desc.", "Alfabética"], index=0)

        df_bar = _count_df(df_filtered["setor"])
        df_bar = df_bar.sort_values(("setor" if order_by=="Alfabética" else "count"),
                                    ascending=False if order_by!="Alfabética" else True)

        fig = px.bar(df_bar, x="setor", y="count",
                     labels={"setor":"Setor","count":"Qtde"},
                     title="Contagem por setor (verdadeiro)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Qtde: %{y:,}")
        fig.update_layout(xaxis_tickangle=25)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Evolução temporal — quantidade por setor")
        stacked = st.checkbox("Empilhar linhas por setor (área empilhada)", value=False)
        tmp = (df_filtered.assign(
                ano_pib=pd.to_numeric(df_filtered["ano_pib"], errors="coerce").astype("Int64"))
               .dropna(subset=["ano_pib"]))
        tmp["ano_pib"] = tmp["ano_pib"].astype(int)
        tmp = (tmp.groupby(["ano_pib","setor"]).size()
               .reset_index(name="qtd").sort_values(["ano_pib","setor"]))
        fig2 = (px.area if stacked else px.line)(
            tmp, x="ano_pib", y="qtd", color="setor",
            template=PLOTLY_TEMPLATE, labels={"ano_pib":"Ano","qtd":"Qtde","setor":"Setor"},
            **({"markers": True} if not stacked else {})
        )
        fig2.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Ano: %{x}<br>Qtde: %{y:,}")
        st.plotly_chart(fig2, use_container_width=True)
        _download(df_filtered, "Baixar recorte (CSV)",
                  f"exploratorio_{'bayes' if src_choice.startswith('Bayes') else 'freq'}.csv")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- Direita: predições + confiança --------
    with right:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        if src_choice.startswith("Bayes") and "pred_setor_bayes" in df_filtered.columns:
            st.subheader("Predições (Bayes)")
            df_pred_bar = _count_df(df_filtered["pred_setor_bayes"]).sort_values("count", ascending=False)
            fig3 = px.bar(df_pred_bar, x="setor", y="count",
                          labels={"setor":"Setor","count":"Qtde"},
                          title="Contagem por setor (predito — Bayes)", template=PLOTLY_TEMPLATE)
            fig3.update_traces(hovertemplate="<b>%{x}</b><br>Qtde: %{y:,}")
            fig3.update_layout(xaxis_tickangle=25)
            st.plotly_chart(fig3, use_container_width=True)

            if "proba_max_bayes" in df_filtered.columns:
                conf = (df_filtered.groupby("pred_setor_bayes")["proba_max_bayes"]
                        .mean().sort_values(ascending=False)).reset_index()
                conf = conf.rename(columns={"pred_setor_bayes":"setor","proba_max_bayes":"proba"})
                fig4 = px.bar(conf, x="setor", y="proba",
                              labels={"setor":"Setor","proba":"Proba média"},
                              title="Probabilidade máxima média por setor (Bayes)",
                              template=PLOTLY_TEMPLATE)
                fig4.update_traces(hovertemplate="<b>%{x}</b><br>Proba média: %{y:.3f}")
                fig4.update_layout(xaxis_tickangle=25, yaxis_range=[0,1])
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("##### Top municípios por confiança (Bayes)")
            topk = st.slider("Top K", 5, 50, 10, 1, key="k_b")
            cols = ["ano_pib","cod_mun","nome_municipio","setor","pred_setor_bayes","proba_max_bayes"]
            st.dataframe(df_filtered.sort_values("proba_max_bayes", ascending=False).head(topk)[cols],
                         hide_index=True, use_container_width=True)

        else:
            st.subheader("Predições (Frequente)")
            df_pred_bar = _count_df(df_filtered["pred_setor_freq"]).sort_values("count", ascending=False)
            fig5 = px.bar(df_pred_bar, x="setor", y="count",
                          labels={"setor":"Setor","count":"Qtde"},
                          title="Contagem por setor (predito — Freq.)", template=PLOTLY_TEMPLATE)
            fig5.update_traces(hovertemplate="<b>%{x}</b><br>Qtde: %{y:,}")
            fig5.update_layout(xaxis_tickangle=25)
            st.plotly_chart(fig5, use_container_width=True)

            if "proba_max_freq" in df_filtered.columns:
                conf = (df_filtered.groupby("pred_setor_freq")["proba_max_freq"]
                        .mean().sort_values(ascending=False)).reset_index()
                conf = conf.rename(columns={"pred_setor_freq":"setor","proba_max_freq":"proba"})
                fig6 = px.bar(conf, x="setor", y="proba",
                              labels={"setor":"Setor","proba":"Proba média"},
                              title="Probabilidade máxima média por setor (Freq.)",
                              template=PLOTLY_TEMPLATE)
                fig6.update_traces(hovertemplate="<b>%{x}</b><br>Proba média: %{y:.3f}")
                fig6.update_layout(xaxis_tickangle=25, yaxis_range=[0,1])
                st.plotly_chart(fig6, use_container_width=True)

            st.markdown("##### Top municípios por confiança (Freq.)")
            topk = st.slider("Top K", 5, 50, 10, 1, key="k_f")
            cols = ["ano_pib","cod_mun","nome_municipio","setor","pred_setor_freq","proba_max_freq"]
            st.dataframe(df_filtered.sort_values("proba_max_freq", ascending=False).head(topk)[cols],
                         hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ====================== PREVISÕES ======================
with tab_prev:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Métricas & Relatórios")

        if src_choice.startswith("Bayes"):
            if df_sum_bayes is not None and len(df_sum_bayes)>0:
                r = df_sum_bayes.iloc[0]
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="small">Cutoff</div><h3>{int(r.get("cutoff", np.nan))}</h3></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="small">Accuracy</div><h3>{float(r.get("accuracy_bayes", np.nan)):.3f}</h3></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="small">F1 Macro</div><h3>{float(r.get("f1_macro_bayes", np.nan)):.3f}</h3></div>', unsafe_allow_html=True)
                _download(df_sum_bayes, "Baixar métricas (CSV)", "metrics_classify_summary_BAYES.csv")
            else:
                st.info("metrics_classify_summary_BAYES.csv não disponível.")

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.subheader("Matriz de confusão (Bayes)")
            if df_cm_bayes is not None and not df_cm_bayes.empty:
                norm = st.selectbox("Normalização", ["Contagem","% por linha (verdadeiro)","% por coluna (predito)","% do total"], 0, key="norm_b")
                cm_show = _normalize_cm(df_cm_bayes, norm)
                ztext = np.vectorize(lambda x: f"{x:.0f}" if norm=="Contagem" else f"{x:.1f}%")(cm_show.values)
                fig = go.Figure(data=go.Heatmap(
                    z=cm_show.values, x=cm_show.columns, y=cm_show.index,
                    colorscale="Blues", colorbar_title=("Qtde" if norm=="Contagem" else "%"),
                    text=ztext, texttemplate="%{text}",
                    hovertemplate="Verdadeiro=%{y}<br>Predito=%{x}<br>Valor=%{z}<extra></extra>"
                ))
                fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10,r=10,t=40,b=10),
                                  title=f"Bayes — Confusion Matrix ({norm})")
                st.plotly_chart(fig, use_container_width=True)
                _download(cm_show.reset_index().rename(columns={"index":"__label__"}),
                          "Baixar matriz (CSV)", "metrics_classify_confusion_matrix_BAYES_view.csv")
            else:
                st.info("metrics_classify_confusion_matrix_BAYES.csv não disponível.")

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.subheader("Relatório de classificação (txt)")
            st.code(txt_bayes or "Relatório não encontrado.", language="text")

        else:
            if df_sum_freq is not None and len(df_sum_freq)>0:
                r = df_sum_freq.iloc[0]
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="small">Cutoff</div><h3>{int(r.get("cutoff", np.nan))}</h3></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="small">Accuracy</div><h3>{float(r.get("accuracy_freq", np.nan)):.3f}</h3></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="small">F1 Macro</div><h3>{float(r.get("f1_macro_freq", np.nan)):.3f}</h3></div>', unsafe_allow_html=True)
                _download(df_sum_freq, "Baixar métricas (CSV)", "metrics_classify_summary.csv")
            else:
                st.info("metrics_classify_summary.csv não disponível.")

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.subheader("Matriz de confusão (Frequente)")
            if df_cm_freq is not None and not df_cm_freq.empty:
                norm = st.selectbox("Normalização", ["Contagem","% por linha (verdadeiro)","% por coluna (predito)","% do total"], 0, key="norm_f")
                cm_show = _normalize_cm(df_cm_freq, norm)
                ztext = np.vectorize(lambda x: f"{x:.0f}" if norm=="Contagem" else f"{x:.1f}%")(cm_show.values)
                fig = go.Figure(data=go.Heatmap(
                    z=cm_show.values, x=cm_show.columns, y=cm_show.index,
                    colorscale="Blues", colorbar_title=("Qtde" if norm=="Contagem" else "%"),
                    text=ztext, texttemplate="%{text}",
                    hovertemplate="Verdadeiro=%{y}<br>Predito=%{x}<br>Valor=%{z}<extra></extra>"
                ))
                fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10,r=10,t=40,b=10),
                                  title=f"Frequente — Confusion Matrix ({norm})")
                st.plotly_chart(fig, use_container_width=True)
                _download(cm_show.reset_index().rename(columns={"index":"__label__"}),
                          "Baixar matriz (CSV)", "metrics_classify_confusion_matrix_freq_view.csv")
            else:
                st.info("metrics_classify_confusion_matrix_freq.csv não disponível.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Previsões por observação (teste)")

        if src_choice.startswith("Bayes"):
            dfshow = df_res_bayes.copy()
            c1,c2,c3,c4 = st.columns(4)
            anos_local = _anos_unicos_as_str(dfshow)
            ano  = c1.selectbox("Ano", ["(todos)"]+anos_local, 0, key="b_ano")
            muni = c2.text_input("Filtro município", key="b_muni")
            true_ = c3.selectbox("Setor verdadeiro", ["(todos)"]+sorted(dfshow["setor"].dropna().unique().tolist()), 0, key="b_true")
            pred_ = c4.selectbox("Setor predito (Bayes)", ["(todos)"]+sorted(dfshow["pred_setor_bayes"].dropna().unique().tolist()), 0, key="b_pred")

            dfshow = _aplicar_filtro_ano(dfshow, ano)
            if muni.strip():
                q = muni.lower().strip()
                dfshow = dfshow[(dfshow["nome_municipio"].astype(str).str.lower().str.contains(q)) |
                                (dfshow["cod_mun"].astype(str).str.contains(q))]
            if true_ != "(todos)": dfshow = dfshow[dfshow["setor"] == true_]
            if pred_ != "(todos)": dfshow = dfshow[dfshow["pred_setor_bayes"] == pred_]

            proba_cols = [c for c in dfshow.columns if c.startswith("p_bayes_mean__")]
            cols = ["ano_pib","cod_mun","nome_municipio","setor","valor","percentual",
                    "pred_setor_bayes","proba_max_bayes","proba_max_bayes_lo","proba_max_bayes_hi"] + proba_cols
            st.dataframe(dfshow[cols], hide_index=True, use_container_width=True)
            _download(dfshow, "Baixar previsões (CSV)", "results_classify_test_BAYES_filtrado.csv")

        else:
            dfshow = df_res_freq.copy()
            c1,c2,c3,c4 = st.columns(4)
            anos_local = _anos_unicos_as_str(dfshow)
            ano  = c1.selectbox("Ano", ["(todos)"]+anos_local, 0, key="f_ano")
            muni = c2.text_input("Filtro município", key="f_muni")
            true_ = c3.selectbox("Setor verdadeiro", ["(todos)"]+sorted(dfshow["setor"].dropna().unique().tolist()), 0, key="f_true")
            pred_ = c4.selectbox("Setor predito (Freq.)", ["(todos)"]+sorted(dfshow["pred_setor_freq"].dropna().unique().tolist()), 0, key="f_pred")

            dfshow = _aplicar_filtro_ano(dfshow, ano)
            if muni.strip():
                q = muni.lower().strip()
                dfshow = dfshow[(dfshow["nome_municipio"].astype(str).str.lower().str.contains(q)) |
                                (dfshow["cod_mun"].astype(str).str.contains(q))]
            if true_ != "(todos)": dfshow = dfshow[dfshow["setor"] == true_]
            if pred_ != "(todos)": dfshow = dfshow[dfshow["pred_setor_freq"] == pred_]

            cols = ["ano_pib","cod_mun","nome_municipio","setor","valor","percentual",
                    "pred_setor_freq","proba_max_freq"]
            st.dataframe(dfshow[cols], hide_index=True, use_container_width=True)
            _download(dfshow, "Baixar previsões (CSV)", "results_classify_test_filtrado.csv")
        st.markdown('</div>', unsafe_allow_html=True)
