# -*- coding: utf-8 -*-
import os, unicodedata
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==================== Sanitização de colunas ====================
_NUMPY_RESERVED = {
    "sum","mean","std","var","min","max","median","average","clip","round","log","log1p","exp",
    "sin","cos","tan","arcsin","arccos","arctan","sqrt","abs","sign","where","all","any","prod",
    "cumsum","cumprod","argsort","argmin","argmax","astype","dtype","shape","size","ndim","ndarray",
    "array","matrix","transpose","dot","einsum","unique","isnan","isfinite","isinf","conj","imag","real",
    "nan","inf","pi","e","load","save","fromfile","tofile"
}
import builtins as _py_builtins
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

# ==================== Aliases ====================
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

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ==================== DB ====================
def get_engine_from_env(args):
    import os
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

# ==================== Normalização / Melt ====================
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

# ==================== IBGE robusto ====================
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

# ==================== Pré-processamento comum ====================
def build_numeric_preprocessor(feat_cols=None):
    # inclui features relativas por município-ano (rank e zscore)
    feat_cols = feat_cols or ["ano_pib","valor_log","percentual","rank_setor_no_mun","zscore_no_mun"]
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feat_cols)
    ], remainder="drop")
    return preproc, feat_cols

def prepare_long_for_classification(long_df):
    dfc = long_df.copy()
    dfc["valor"] = pd.to_numeric(dfc["valor"], errors="coerce").clip(lower=0)
    dfc["percentual"] = pd.to_numeric(dfc["percentual"], errors="coerce")
    dfc["valor_log"] = np.log1p(dfc["valor"])

    # Sinais relativos por município-ano (100% numéricos)
    grp = dfc.groupby(["ano_pib","cod_mun"], group_keys=False)
    dfc["rank_setor_no_mun"] = grp["valor"].rank(ascending=False, method="dense")
    dfc["zscore_no_mun"] = grp["valor"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))

    dfc = dfc.replace([np.inf, -np.inf], np.nan)

    X = dfc[["ano_pib","valor_log","percentual","rank_setor_no_mun","zscore_no_mun"]]
    y = dfc["setor"].astype(str)
    return X, y, dfc
