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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import builtins as _py_builtins

# -------------------- Padronização / Sanitização --------------------
_NUMPY_RESERVED = {
    "sum","mean","std","var","min","max","median","average","clip","round","log","log1p","exp",
    "sin","cos","tan","arcsin","arccos","arctan","sqrt","abs","sign","where","all","any","prod",
    "cumsum","cumprod","argsort","argmin","argmax","astype","dtype","shape","size","ndim","ndarray",
    "array","matrix","transpose","dot","einsum","unique","isnan","isfinite","isinf","conj","imag","real",
    "nan","inf","pi","e","load","save","fromfile","tofile"
}
_BUILTIN_RESERVED = {n for n in dir(_py_builtins)}
_RESERVED = {s.lower() for s in (_NUMPY_RESERVED | _BUILTIN_RESERVED)}

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        base = str(c).strip().lower().replace(" ", "_")
        base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
        if base in _RESERVED:
            base = f"{base}_col"
        new_cols.append(base)
    out = df.copy()
    out.columns = new_cols
    return out

# -------------------- ALIASES --------------------
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

def _choose(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# -------------------- DB --------------------
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
    return sanitize_columns(df)

# -------------------- Normalização / Melt --------------------
def normalize_keys(df):
    cols = df.columns
    c_ano = _choose(cols, ALIASES["ano_pib"]); c_cod = _choose(cols, ALIASES["cod_mun"])
    if not c_ano or not c_cod:
        raise ValueError(f"faltou ano/cod_mun. Colunas: {list(cols)}")

    df_keys = df[[c_ano, c_cod]].copy().rename(columns={c_ano:"ano_pib", c_cod:"cod_mun"})
    df_keys["ano_pib"] = pd.to_numeric(df_keys["ano_pib"], errors="coerce").astype("Int64")
    df_keys = df_keys.dropna(subset=["ano_pib","cod_mun"]).copy()
    df_keys["ano_pib"] = df_keys["ano_pib"].astype(int)
    df_keys["cod_mun"] = df_keys["cod_mun"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(7)
    df_keys["nome_municipio"] = df_keys["cod_mun"]

    keep_cols = [c for c in df.columns if c not in df_keys.columns]
    return df_keys.merge(df[keep_cols], left_index=True, right_index=True, how="left")

def find_sector_columns(df):
    cols = df.columns
    found = {}
    for canon in SETORES_CANON:
        c = _choose(cols, ALIASES[canon])
        if c:
            found[canon] = c
    return found

def melt_setores(df):
    mapping = find_sector_columns(df)
    if not mapping:
        raise ValueError("sem colunas de setores mapeadas (após sanitize)")
    value_cols = list(mapping.values())
    id_vars = [c for c in ["ano_pib","cod_mun","nome_municipio"] if c in df.columns]

    df_long = df[id_vars + value_cols].melt(
        id_vars=id_vars, value_vars=value_cols, var_name="setor", value_name="valor"
    )
    inv = {v:k for k,v in mapping.items()}
    df_long["setor"] = df_long["setor"].map(inv).fillna(df_long["setor"])

    df_total = (
        df_long.groupby(["ano_pib","cod_mun"], as_index=False)["valor"]
        .sum()
        .rename(columns={"valor":"total_setores"})
    )
    df_long = df_long.merge(df_total, on=["ano_pib","cod_mun"], how="left")
    df_long["percentual"] = np.where(
        df_long["total_setores"]>0, 100.0*df_long["valor"]/df_long["total_setores"], np.nan
    )
    return df_long

# -------------------- IBGE robusto --------------------
def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        base = _strip_accents(str(c)).lower().strip().replace(" ", "_")
        base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
        cols.append(base)
    out = df.copy()
    out.columns = cols
    return out

def try_merge_ibge(df, xls_path):
    if not xls_path or not os.path.exists(xls_path):
        print("[INFO] sem IBGE XLS/XLSX — pulando")
        return df

    ext = os.path.splitext(xls_path)[1].lower()
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"

    try:
        x = pd.read_excel(xls_path, header=6, engine=engine)
    except Exception:
        x = pd.read_excel(xls_path, header=0, engine=engine)

    x = _normalize_cols(x)

    code_candidates = [
        "codigo_municipio_completo","codigo_municipio","cod_municipio",
        "codigo_municipio_ibge","codigo_ibge","codigo_mun","cod_ibge"
    ]
    name_candidates = [
        "nome_municipio","nome_municipio_","nome_municipio_ibge","nome_do_municipio","nome"
    ]
    code_col = next((c for c in code_candidates if c in x.columns), None)
    name_col = next((c for c in name_candidates if c in x.columns), None)

    if not code_col or not name_col:
        try:
            x2 = pd.read_excel(xls_path, header=None, engine=engine)
            header_row = None
            for i in range(min(20, len(x2))):
                row_vals = [_strip_accents(str(v)).lower().strip() for v in x2.iloc[i].values]
                if ("codigo municipio completo" in row_vals) or ("codigo_municipio_completo" in row_vals):
                    header_row = i; break
            if header_row is not None:
                x = pd.read_excel(xls_path, header=header_row, engine=engine)
                x = _normalize_cols(x)
                code_col = next((c for c in code_candidates if c in x.columns), None)
                name_col = next((c for c in name_candidates if c in x.columns), None)
        except Exception:
            pass

    if not code_col or not name_col:
        print(f"[WARN] IBGE: não encontrei colunas de código/nome. Colunas: {list(x.columns)[:10]}...")
        return df

    x = x[[code_col, name_col]].dropna()
    x.rename(columns={code_col:"codigo_municipio_completo", name_col:"nome_municipio_ibge"}, inplace=True)
    x["codigo_municipio_completo"] = (
        x["codigo_municipio_completo"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(7)
    )

    before = len(df)
    df_merged = df.merge(x, left_on="cod_mun", right_on="codigo_municipio_completo", how="left")
    df_merged["nome_municipio"] = df_merged.get("nome_municipio", df_merged["cod_mun"])
    df_merged["nome_municipio"] = df_merged["nome_municipio_ibge"].fillna(df_merged["nome_municipio"])
    matched = df_merged["nome_municipio_ibge"].notna().sum()
    print(f"[INFO] IBGE merge: {matched}/{before} códigos casados ({matched/before:.1%}).")

    return df_merged.drop(columns=["codigo_municipio_completo","nome_municipio_ibge"], errors="ignore")

# -------------------- Pipeline frequente --------------------
def build_classifier_pipeline():
    feat_cols = ["ano_pib","valor_log","percentual"]
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feat_cols)
    ], remainder="drop")
    clf = LogisticRegression(multi_class="multinomial", class_weight="balanced",
                             max_iter=1000, solver="lbfgs")
    return Pipeline([("pre", preproc), ("clf", clf)])

def prepare_long_for_classification(long_df):
    dfc = long_df.copy()
    dfc["valor"] = pd.to_numeric(dfc["valor"], errors="coerce").clip(lower=0)
    dfc["percentual"] = pd.to_numeric(dfc["percentual"], errors="coerce")
    dfc["valor_log"] = np.log1p(dfc["valor"])
    dfc = dfc.replace([np.inf, -np.inf], np.nan)
    X = dfc[["ano_pib","valor_log","percentual"]]
    y = dfc["setor"].astype(str)
    return X, y, dfc

# -------------------- Bayes (multiclasse, softmax + ADVI) --------------------
def fit_bayes_softmax(preprocessor, Xtr, ytr_str, Xte, ci=0.90, draws=1000, advi_steps=15000):
    try:
        import pymc as pm
        try:
            import pytensor.tensor as pt  # PyMC 5
        except Exception:
            import aesara.tensor as pt   # fallback PyMC 4
        import arviz as az
    except Exception as e:
        return False, f"PyMC indisponível: {e}", None, None, None, None

    Xtr_p = preprocessor.transform(Xtr)
    Xte_p = preprocessor.transform(Xte)
    classes = np.array(sorted(pd.Series(ytr_str).unique()))
    K = len(classes)
    if K < 3:
        return False, "Softmax bayesiano: requer K>=3 classes.", None, None, None, None

    y_idx = pd.Series(ytr_str).map({c:i for i,c in enumerate(classes)}).values.astype("int64")
    nfeat = Xtr_p.shape[1]
    alpha = (1.0 - ci)/2.0
    p_lo = 100*alpha
    p_hi = 100*(1-alpha)

    try:
        with pm.Model() as model:
            # priors CONSERVADORES
            B  = pm.Normal("B",  mu=0.0, sigma=1.0, shape=(nfeat, K-1))
            b0 = pm.Normal("b0", mu=0.0, sigma=1.0, shape=(K-1,))
            eta_km1 = pt.dot(Xtr_p, B) + b0
            eta = pt.concatenate([eta_km1, pt.zeros((Xtr_p.shape[0],1))], axis=1)
            p_tr = pm.Deterministic("p_tr", pm.math.softmax(eta))
            pm.Categorical("y_obs", p=p_tr, observed=y_idx)

            approx = pm.fit(
                n=advi_steps,
                method="advi",
                obj_optimizer=pm.adam(learning_rate=1e-3),
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='absolute')],
                progressbar=False
            )
            idata = approx.sample(draws=draws)

        B_s  = np.asarray(idata.posterior["B"]).reshape(-1, nfeat, K-1)
        b0_s = np.asarray(idata.posterior["b0"]).reshape(-1, K-1)
        S = B_s.shape[0]
        logits_km1 = np.einsum("sfk,nf->snk", B_s, Xte_p) + b0_s[:,None,:]
        logits = np.concatenate([logits_km1, np.zeros((S, Xte_p.shape[0], 1))], axis=2)
        ex = np.exp(logits - logits.max(axis=2, keepdims=True))
        probs = ex / ex.sum(axis=2, keepdims=True)   # [S, N, K]

        proba_mean = probs.mean(axis=0)
        proba_lo   = np.percentile(probs, p_lo, axis=0)
        proba_hi   = np.percentile(probs, p_hi, axis=0)
        return True, None, classes, proba_mean, proba_lo, proba_hi
    except Exception as e:
        return False, f"Falha PyMC (ADVI): {e}", None, None, None, None

# -------------------- Main --------------------
def main():
    argp = argparse.ArgumentParser(description="Classificação de SETOR (frequente + bayesiana) com IC.")
    # DB
    argp.add_argument("--db_host", default=None); argp.add_argument("--db_port", default=None)
    argp.add_argument("--db_name", default=None); argp.add_argument("--db_user", default=None)
    argp.add_argument("--db_pwd",  default=None); argp.add_argument("--db_schema", default=None)
    argp.add_argument("--table", default="public.pib_municipios")
    # I/O
    argp.add_argument("--out_dir", default="/content/drive/MyDrive/PIB_Forecast")
    argp.add_argument("--ibge_xls_path", default="/content/drive/MyDrive/Bases de Dados/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls")
    # split
    argp.add_argument("--cutoff_year", type=int, default=None)
    # bayes (multiclasse)
    argp.add_argument("--use_bayes", action="store_true")
    argp.add_argument("--bayes_draws", type=int, default=1000)
    argp.add_argument("--advi_steps", type=int, default=15000)
    argp.add_argument("--ci", type=float, default=0.90)
    args, _ = argp.parse_known_args()

    engine = get_engine_from_env(args)
    out_dir = _ensure_dir(args.out_dir)

    print("[INFO] Carregando tabela..."); df_raw = load_raw_table(engine, args.table)
    print("[INFO] Normalizando chaves..."); df_norm = normalize_keys(df_raw)
    df_norm = try_merge_ibge(df_norm, args.ibge_xls_path)
    print("[INFO] Melt & percentuais..."); df_long = melt_setores(df_norm)

    # split temporal
    if args.cutoff_year is None:
        anos_sorted = np.sort(df_long["ano_pib"].unique())
        if len(anos_sorted) < 2:
            raise ValueError("Poucos anos")
        cutoff = int(anos_sorted[-2])
    else:
        cutoff = int(args.cutoff_year)

    df_train = df_long[df_long["ano_pib"] <= cutoff].copy()
    df_test  = df_long[df_long["ano_pib"] >  cutoff].copy()
    if df_train.empty or df_test.empty:
        raise ValueError("Split vazio; ajuste --cutoff_year")

    Xtr, ytr, _   = prepare_long_for_classification(df_train)
    Xte, yte, dte = prepare_long_for_classification(df_test)

    # frequente
    print("[INFO] Treinando FREQUENTISTA...")
    pipe = build_classifier_pipeline().fit(Xtr, ytr)
    preproc = pipe.named_steps["pre"]
    ypred_freq = pipe.predict(Xte)
    yproba_freq = pipe.predict_proba(Xte)
    labels_freq = pipe.classes_
    acc_freq = accuracy_score(yte, ypred_freq)
    f1m_freq = f1_score(yte, ypred_freq, average="macro")
    cm_freq = confusion_matrix(yte, ypred_freq, labels=labels_freq)

    # bayes (multiclasse)
    bayes_ok=False; acc_bayes=f1m_bayes=None
    if args.use_bayes:
        print("[INFO] Treinando BAYESIANO (multiclasse, softmax+ADVI)...")
        ok_flag, msg_bayes, classes_b, p_mean_b, p_lo_b, p_hi_b = fit_bayes_softmax(
            preprocessor=preproc, Xtr=Xtr, ytr_str=ytr, Xte=Xte,
            ci=args.ci, draws=args.bayes_draws, advi_steps=args.advi_steps
        )
        if not ok_flag:
            print(f"[WARN] Bayes falhou: {msg_bayes}")
        else:
            bayes_ok=True
            ypred_bayes = [classes_b[i] for i in p_mean_b.argmax(axis=1)]
            acc_bayes = accuracy_score(yte, ypred_bayes)
            f1m_bayes = f1_score(yte, ypred_bayes, average="macro")
            cm_bayes = confusion_matrix(yte, ypred_bayes, labels=classes_b)
            pd.DataFrame(cm_bayes, index=classes_b, columns=classes_b)\
              .to_csv(os.path.join(out_dir, "metrics_classify_confusion_matrix_bayes.csv"), encoding="utf-8")

    # resultados
    df_res = dte[["ano_pib","cod_mun","nome_municipio","setor","valor","percentual"]].copy()
    df_res["pred_setor_freq"] = ypred_freq
    df_res["proba_max_freq"]  = yproba_freq.max(axis=1)
    if bayes_ok:
        for j,cname in enumerate(classes_b):
            df_res[f"p_bayes_mean__{cname}"] = p_mean_b[:,j]
            df_res[f"p_bayes_lo__{cname}"]   = p_lo_b[:,j]
            df_res[f"p_bayes_hi__{cname}"]   = p_hi_b[:,j]
        idx_best = p_mean_b.argmax(axis=1)
        df_res["pred_setor_bayes"] = [classes_b[i] for i in idx_best]
        df_res["proba_max_bayes"]  = p_mean_b.max(axis=1)
        df_res["proba_max_bayes_lo"] = p_lo_b[np.arange(len(idx_best)), idx_best]
        df_res["proba_max_bayes_hi"] = p_hi_b[np.arange(len(idx_best)), idx_best]

    df_res.to_csv(os.path.join(out_dir, "results_classify_test.csv"), index=False, encoding="utf-8")

    # métricas & confusões
    pd.DataFrame([{
        "cutoff": cutoff,
        "accuracy_freq": acc_freq, "f1_macro_freq": f1m_freq,
        **({ "accuracy_bayes": acc_bayes, "f1_macro_bayes": f1m_bayes } if bayes_ok else {})
    }]).to_csv(os.path.join(out_dir, "metrics_classify_summary.csv"), index=False, encoding="utf-8")

    pd.DataFrame(cm_freq, index=labels_freq, columns=labels_freq)\
      .to_csv(os.path.join(out_dir, "metrics_classify_confusion_matrix_freq.csv"), encoding="utf-8")

    print(f"[OK] FREQ  -> acc={acc_freq:.3f} | f1={f1m_freq:.3f}")
    if bayes_ok:
        print(f"[OK] BAYES -> acc={acc_bayes:.3f} | f1={f1m_bayes:.3f}")
    else:
        print("[INFO] Bayes não rodou (sem --use_bayes ou PyMC indisponível).")

if __name__ == "__main__":
    main()
