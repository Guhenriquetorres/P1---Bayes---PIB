# -*- coding: utf-8 -*-
import os, warnings, argparse, unicodedata
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ========== Utils básicos ==========
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def get_engine_from_env(args):
    host = args.db_host or os.getenv("IESB_HOST","bigdata.dataiesb.com")
    port = int(args.db_port or os.getenv("IESB_PORT","5432"))
    db   = args.db_name or os.getenv("IESB_DB","iesb")
    usr  = args.db_user or os.getenv("IESB_USER","data_iesb")
    pwd  = args.db_pwd  or os.getenv("IESB_PWD","iesb")
    sch  = args.db_schema or os.getenv("IESB_SCHEMA","public")
    url  = f"postgresql+psycopg2://{usr}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, connect_args={"options": f"-csearch_path={sch}"})

def load_raw_table(eng, table):
    df = pd.read_sql(text(f"SELECT * FROM {table}"), con=eng)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ========== Normalização / IBGE ==========
ALIASES = {
    "ano_pib": ["ano_pib","no_pib","ano","anopib","ano_referencia","ano_ref"],
    "cod_mun": ["cod_mun","codigo_municipio_dv","codigo_municipio","cd_municipio","co_municipio",
                "co_mun","id_municipio","cod_ibge","codigo_ibge","codigo_mun","cod_municipio"],
    "vl_agropecuaria": ["vl_agropecuaria","agropecuaria","valor_agropecuaria"],
    "vl_industria":    ["vl_industria","industria","valor_industria"],
    "vl_servicos":     ["vl_servicos","servicos","valor_servicos"],
    "vl_administracao":["vl_administracao","administracao","adm_publica","valor_administracao"],
    "vl_subsidios":    ["vl_subsidios","subsidios","valor_subsidios"],
}
SETORES_CANON = ["vl_agropecuaria","vl_industria","vl_servicos","vl_administracao","vl_subsidios"]

def _choose(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def normalize_keys(df):
    cols = df.columns
    c_ano = _choose(cols, ALIASES["ano_pib"]); c_cod = _choose(cols, ALIASES["cod_mun"])
    if not c_ano or not c_cod:
        raise ValueError(f"faltou ano/cod_mun. Colunas: {list(cols)}")
    out = df[[c_ano,c_cod]].copy().rename(columns={c_ano:"ano_pib", c_cod:"cod_mun"})
    out["ano_pib"] = pd.to_numeric(out["ano_pib"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ano_pib","cod_mun"]).copy()
    out["ano_pib"] = out["ano_pib"].astype(int)
    out["cod_mun"] = out["cod_mun"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(7)
    out["nome_municipio"] = out["cod_mun"]
    keep = [c for c in df.columns if c not in out.columns]
    return out.merge(df[keep], left_index=True, right_index=True, how="left")

def _strip_accents(s):
    if not isinstance(s, str): s = str(s)
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))

def _normalize_cols(df):
    cols = []
    for c in df.columns:
        base = _strip_accents(str(c)).lower().strip().replace(" ","_")
        base = "".join(ch if ch.isalnum() or ch=="_" else "_" for ch in base)
        cols.append(base)
    df2 = df.copy(); df2.columns = cols; return df2

def try_merge_ibge(df, xls_path):
    if not xls_path or not os.path.exists(xls_path):
        print("[INFO] sem IBGE XLS/XLSX — pulando"); return df
    ext = os.path.splitext(xls_path)[1].lower()
    engine = "openpyxl" if ext==".xlsx" else "xlrd"
    try:
        x = pd.read_excel(xls_path, header=6, engine=engine)
    except Exception:
        x = pd.read_excel(xls_path, header=0, engine=engine)
    x = _normalize_cols(x)
    code_candidates = ["codigo_municipio_completo","codigo_municipio","cod_municipio",
                       "codigo_municipio_ibge","codigo_ibge","codigo_mun","cod_ibge"]
    name_candidates = ["nome_municipio","nome_municipio_","nome_municipio_ibge","nome_do_municipio","nome"]
    code_col = next((c for c in code_candidates if c in x.columns), None)
    name_col = next((c for c in name_candidates if c in x.columns), None)
    if not code_col or not name_col:
        print(f"[WARN] IBGE: não encontrei colunas. Colunas={list(x.columns)[:8]}"); return df
    x = x[[code_col,name_col]].dropna().rename(columns={code_col:"codigo_municipio_completo", name_col:"nome_municipio_ibge"})
    x["codigo_municipio_completo"] = x["codigo_municipio_completo"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(7)
    dfm = df.merge(x, left_on="cod_mun", right_on="codigo_municipio_completo", how="left")
    dfm["nome_municipio"] = dfm.get("nome_municipio", dfm["cod_mun"])
    dfm["nome_municipio"] = dfm["nome_municipio_ibge"].fillna(dfm["nome_municipio"])
    return dfm.drop(columns=["codigo_municipio_completo","nome_municipio_ibge"], errors="ignore")

def find_sector_columns(df):
    found = {}
    for s in SETORES_CANON:
        c = _choose(df.columns, ALIASES[s])
        if c: found[s] = c
    return found

# *** NOVO: robusto a negativos e percentuais absurdos
def melt_setores(df):
    mapping = find_sector_columns(df)
    if not mapping: raise ValueError("sem colunas de setores mapeadas")
    value_cols = list(mapping.values())
    id_vars = [c for c in ["ano_pib","cod_mun","nome_municipio"] if c in df.columns]

    # valores numéricos e zera negativos
    vals = df[value_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    vals_pos = vals.where(vals >= 0.0, other=0.0)

    # total apenas do positivo
    total_pos = vals_pos.sum(axis=1).astype("float64")

    # reconstrói df fixo e derrete
    df_fixed = pd.concat([df[id_vars].reset_index(drop=True), vals_pos.reset_index(drop=True)], axis=1)
    long = df_fixed.melt(id_vars=id_vars, value_vars=value_cols, var_name="setor", value_name="valor")
    inv = {v:k for k,v in mapping.items()}
    long["setor"] = long["setor"].map(inv).fillna(long["setor"])

    # injeta total e calcula percentual seguro
    reps = len(value_cols)
    long["total_setores"] = np.repeat(total_pos.values, repeats=reps).astype("float64")
    long["percentual"] = np.where(long["total_setores"] > 0.0, 100.0*long["valor"]/long["total_setores"], np.nan)
    long["percentual"] = long["percentual"].clip(lower=0.0, upper=100.0)

    return long

# ========== Pré-processador ==========
def build_numeric_preprocessor():
    feat_cols = ["ano_pib","valor_log","percentual"]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), feat_cols)
    ], remainder="drop")
    return pre, feat_cols

def prepare_long_for_classification(long_df):
    dfc = long_df.copy()
    dfc["valor"] = pd.to_numeric(dfc["valor"], errors="coerce").clip(lower=0)  # já vem >=0 do melt robusto
    dfc["percentual"] = pd.to_numeric(dfc["percentual"], errors="coerce").clip(lower=0, upper=100)
    dfc["valor_log"] = np.log1p(dfc["valor"])
    dfc = dfc.replace([np.inf,-np.inf], np.nan)
    X = dfc[["ano_pib","valor_log","percentual"]]
    y = dfc["setor"].astype(str)
    return X, y, dfc

# ========== BAYES (só VI/ADVI — SEM NUTS) ==========
def fit_bayes_vi(preprocessor, Xtr, ytr_str, Xte, ci=0.90, draws=1000, advi_steps=20000, lr=5e-4):
    """
    - K==2: logística binária com VI.
    - K>=3: softmax (multiclasse) com ADVI.
    SEM NUTS.
    """
    try:
        import pymc as pm
        try:
            import pytensor.tensor as pt
        except Exception:
            import aesara.tensor as pt
        import arviz as az  # só para checagens internas (não obrigatório)
    except Exception as e:
        return False, f"PyMC indisponível: {e}", None, None, None, None

    Xtr_p = preprocessor.transform(Xtr)
    Xte_p = preprocessor.transform(Xte)

    # sanity antes de treinar
    if np.isnan(Xtr_p).any() or np.isinf(Xtr_p).any():
        return False, "Xtr_p contém NaN/Inf após preprocess.", None, None, None, None
    if np.isnan(Xte_p).any() or np.isinf(Xte_p).any():
        return False, "Xte_p contém NaN/Inf após preprocess.", None, None, None, None

    classes = np.array(sorted(pd.Series(ytr_str).astype(str).unique()))
    K = len(classes)
    alpha = (1.0 - ci)/2.0
    plo, phi = 100*alpha, 100*(1-alpha)
    nfeat = Xtr_p.shape[1]

    if K == 2:
        # ---- Binário via VI ----
        y_bin = (pd.Series(ytr_str).astype(str).values == classes[1]).astype("int8")
        with pm.Model() as m:
            coefs = pm.Normal("coefs", 0, 1.0, shape=nfeat)
            intercept = pm.Normal("intercept", 0, 1.0)
            logits = intercept + pt.dot(Xtr_p, coefs)
            p = pm.math.sigmoid(logits)
            pm.Bernoulli("y_obs", p=p, observed=y_bin)
            approx = pm.fit(
                n=advi_steps, method="advi",
                obj_optimizer=pm.adam(learning_rate=lr),
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="absolute")],
                progressbar=False
            )
            idata = approx.sample(draws=draws)

        coefs_s = np.asarray(idata.posterior["coefs"]).reshape(-1, nfeat)
        inter_s = np.asarray(idata.posterior["intercept"]).reshape(-1)
        logits_te = (Xte_p @ coefs_s.T) + inter_s
        pte = 1/(1+np.exp(-logits_te))
        p_mean = pte.mean(axis=1); p_lo = np.percentile(pte, plo, axis=1); p_hi = np.percentile(pte, phi, axis=1)
        proba_mean = np.vstack([1-p_mean, p_mean]).T
        proba_lo   = np.vstack([1-p_hi,   p_lo  ]).T
        proba_hi   = np.vstack([1-p_lo,   p_hi  ]).T
        return True, None, classes, proba_mean, proba_lo, proba_hi

    # ---- Multiclasse (Softmax) via ADVI ----
    y_idx = pd.Series(ytr_str).astype(str).map({c:i for i,c in enumerate(classes)}).values.astype("int64")
    with pm.Model() as m:
        # priors conservadores ajudam a estabilidade
        B  = pm.Normal("B", 0, 1.0, shape=(nfeat, K-1))
        b0 = pm.Normal("b0",0, 1.0, shape=(K-1,))
        eta_km1 = Xtr_p @ B + b0
        eta = pm.math.concatenate([eta_km1, pm.math.zeros((Xtr_p.shape[0],1))], axis=1)  # classe de referência
        p_tr = pm.math.softmax(eta)
        pm.Categorical("y_obs", p=p_tr, observed=y_idx)

        approx = pm.fit(
            n=advi_steps, method="advi",
            obj_optimizer=pm.adam(learning_rate=lr),
            callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="absolute")],
            progressbar=False
        )
        idata = approx.sample(draws=draws)

    B_s  = np.asarray(idata.posterior["B"]).reshape(-1, nfeat, K-1)
    b0_s = np.asarray(idata.posterior["b0"]).reshape(-1, K-1)
    S = B_s.shape[0]
    logits_km1 = np.einsum("sfk,nf->snk", B_s, Xte_p) + b0_s[:,None,:]
    logits = np.concatenate([logits_km1, np.zeros((S, Xte_p.shape[0],1))], axis=2)
    ex = np.exp(logits - logits.max(axis=2, keepdims=True))
    probs = ex / ex.sum(axis=2, keepdims=True)

    proba_mean = probs.mean(axis=0)
    proba_lo   = np.percentile(probs, plo, axis=0)
    proba_hi   = np.percentile(probs, phi, axis=0)
    return True, None, classes, proba_mean, proba_lo, proba_hi

# ========== MAIN ==========
def main():
    p = argparse.ArgumentParser(description="Classificação de SETOR — Bayes (VI/ADVI, sem NUTS).")
    # DB
    p.add_argument("--db_host", default=None); p.add_argument("--db_port", default=None)
    p.add_argument("--db_name", default=None); p.add_argument("--db_user", default=None)
    p.add_argument("--db_pwd",  default=None); p.add_argument("--db_schema", default=None)
    p.add_argument("--table", default="public.pib_municipios")
    # I/O
    p.add_argument("--out_dir", default="/content/drive/MyDrive/PIB_Forecast")
    p.add_argument("--ibge_xls_path", default="/content/drive/MyDrive/Bases de Dados/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls")
    # split
    p.add_argument("--cutoff_year", type=int, default=None)
    # bayes
    p.add_argument("--bayes_draws", type=int, default=1000)
    p.add_argument("--advi_steps", type=int, default=20000)
    p.add_argument("--ci", type=float, default=0.90)
    args, _ = p.parse_known_args()

    eng = get_engine_from_env(args)
    out_dir = ensure_dir(args.out_dir)

    print("[INFO] Carregando tabela..."); df_raw = load_raw_table(eng, args.table)
    print("[INFO] Normalizando chaves..."); df_norm = normalize_keys(df_raw)
    df_norm = try_merge_ibge(df_norm, args.ibge_xls_path)
    print("[INFO] Melt & percentuais..."); df_long = melt_setores(df_norm)

    # split temporal
    anos = np.sort(df_long["ano_pib"].unique())
    if args.cutoff_year is None:
        if len(anos)<2: raise ValueError("Poucos anos")
        cutoff = int(anos[-2])
    else:
        cutoff = int(args.cutoff_year)

    train = df_long[df_long["ano_pib"] <= cutoff].copy()
    test  = df_long[df_long["ano_pib"] >  cutoff].copy()
    if train.empty or test.empty: raise ValueError("Split vazio; ajuste --cutoff_year")

    Xtr, ytr, _   = prepare_long_for_classification(train)
    Xte, yte, dte = prepare_long_for_classification(test)

    # pré-processador numérico (fit no treino)
    pre, _ = build_numeric_preprocessor()
    pre = pre.fit(Xtr)

    # ----- Bayes (SEM NUTS) -----
    print("[INFO] Treinando BAYES (VI/ADVI, sem NUTS)...")
    ok, msg, classes_b, p_mean_b, p_lo_b, p_hi_b = fit_bayes_vi(
        preprocessor=pre, Xtr=Xtr, ytr_str=ytr, Xte=Xte,
        ci=args.ci, draws=args.bayes_draws, advi_steps=args.advi_steps, lr=5e-4
    )
    if not ok:
        print(f"[ERRO] Bayes falhou: {msg}"); return

    ypred = [classes_b[i] for i in p_mean_b.argmax(axis=1)]
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    cm  = confusion_matrix(yte, ypred, labels=classes_b)

    # salvar resultados
    res = dte[["ano_pib","cod_mun","nome_municipio","setor","valor","percentual"]].copy()
    for j,c in enumerate(classes_b):
        res[f"p_bayes_mean__{c}"] = p_mean_b[:,j]
        res[f"p_bayes_lo__{c}"]   = p_lo_b[:,j]
        res[f"p_bayes_hi__{c}"]   = p_hi_b[:,j]
    idx = p_mean_b.argmax(axis=1)
    res["pred_setor_bayes"] = [classes_b[i] for i in idx]
    res["proba_max_bayes"]  = p_mean_b.max(axis=1)
    res["proba_max_bayes_lo"] = p_lo_b[np.arange(len(idx)), idx]
    res["proba_max_bayes_hi"] = p_hi_b[np.arange(len(idx)), idx]

    res.to_csv(os.path.join(out_dir, "results_classify_test_BAYES.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"cutoff": cutoff, "accuracy_bayes": acc, "f1_macro_bayes": f1m}])\
      .to_csv(os.path.join(out_dir, "metrics_classify_summary_BAYES.csv"), index=False, encoding="utf-8")
    pd.DataFrame(cm, index=classes_b, columns=classes_b)\
      .to_csv(os.path.join(out_dir, "metrics_classify_confusion_matrix_BAYES.csv"), encoding="utf-8")

    print(f"[OK] BAYES (sem NUTS) -> acc={acc:.3f} | f1={f1m:.3f}")

if __name__ == "__main__":
    main()
