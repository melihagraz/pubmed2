import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,  # altta SafeSelectKBest ile sarıyoruz
    RFE,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import get_scorer
from pubmed_searcher import SimplePubMedSearcher
import time, json, sqlite3, warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Safe SelectKBest ----------------
class SafeSelectKBest(SelectKBest):
    """k > n_features olduğunda k'yı otomatik güvenli seviyeye çeker."""
    def fit(self, X, y=None):
        n_features = X.shape[1] if hasattr(X, "shape") else np.asarray(X).shape[1]
        if isinstance(self.k, int) and self.k > n_features:
            self.k = n_features
        elif isinstance(self.k, int) and self.k < 1:
            self.k = 1
        return super().fit(X, y)
# -------------------------------------------------

@dataclass
class TrialPlan:
    strategy: str
    params: Dict[str, Any]
    comment: str = ""

@dataclass
class TrialResult:
    metric_name: str
    metric_value: float
    metric_std: float
    n_features: int
    selected_features: List[str]
    pipeline_repr: str
    duration_sec: float
    reflection: str = ""

@dataclass
class AgentConfig:
    target_metric: str = "roc_auc"
    target_threshold: Optional[float] = None
    budget_trials: int = 30
    budget_seconds: Optional[int] = None
    cv_splits: int = 5
    random_state: int = 42
    enable_optuna: bool = True
    optuna_timeout_per_trial: Optional[int] = 60
    imbalance_threshold: float = 0.15
    hitl_enabled: bool = False
    hitl_auto_blocklist: List[str] = None

links = ["", ""]

class ExperimentStore:
    def __init__(self, db_path: str = "agent_runs.sqlite"):
        self.db_path = db_path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, plan TEXT, result TEXT)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS artifacts (
                key TEXT PRIMARY KEY, value TEXT)"""
        )
        con.commit(); con.close()

    def log_trial(self, plan: TrialPlan, result: TrialResult):
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("INSERT INTO trials (ts, plan, result) VALUES (?, ?, ?)",
                    (time.time(), json.dumps(asdict(plan)), json.dumps(asdict(result))))
        con.commit(); con.close()

    def save_artifact(self, key: str, value: Dict[str, Any]):
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("REPLACE INTO artifacts (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)))
        con.commit(); con.close()

    def load_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("SELECT value FROM artifacts WHERE key=?", (key,))
        row = cur.fetchone(); con.close()
        return json.loads(row[0]) if row else None

    def dataframe(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trials ORDER BY id ASC", con)
        con.close()
        if not df.empty:
            df["plan"] = df["plan"].apply(json.loads)
            df["result"] = df["result"].apply(json.loads)
        return df

class HumanInTheLoop:
    def __init__(self, enabled: bool = False, auto_blocklist: Optional[List[str]] = None):
        self.enabled = enabled
        self.auto_blocklist = auto_blocklist or []
    def approve_features(self, selected: List[str]) -> List[str]:
        if not self.enabled: return selected
        return [f for f in selected if not any(b in f for b in self.auto_blocklist)]

class LiteratureEnhancedAgent:
    def __init__(
        self,
        config: AgentConfig,
        pubmed_searcher: Optional[SimplePubMedSearcher] = None,
        store: Optional[ExperimentStore] = None,
        disease_context: Optional[str] = None,
        hitl: Optional[HumanInTheLoop] = None,
    ):
        self.cfg = config
        self.disease_context = disease_context
        self.pubmed_searcher = pubmed_searcher
        self.store = store or ExperimentStore()
        self.hitl = hitl or HumanInTheLoop(config.hitl_enabled, config.hitl_auto_blocklist)
        self.best_pipeline: Optional[Pipeline] = None
        self.best_score: float = -np.inf
        self.best_features: List[str] = []
        self.task_is_classification: Optional[bool] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.history: List[Tuple[TrialPlan, TrialResult]] = []
        self.literature_cache: Dict[str, dict] = {}

    # -------- helpers: y temizleme + OHE uyumluluğu ----------
    @staticmethod
    def _clean_target(y: pd.Series) -> pd.Series:
        """Y'deki NaN'ları ele ve yaygın ikili string etiketleri 0/1'e map'le."""
        # strip & lower map
        def _norm(v):
            try: return str(v).strip().lower()
            except: return v
        uniques = pd.unique(y.dropna().map(_norm))
        mapping = None
        # yaygın ikililer
        pairs = [
            ({"m","malignant"},{"b","benign"}),
            ({"yes","y","1"},{"no","n","0"}),
            ({"true","1"},{"false","0"}),
        ]
        for pos, neg in pairs:
            if set(uniques) <= (pos|neg) and len(set(uniques)&pos)>0 and len(set(uniques)&neg)>0:
                mapping = {**{p:1 for p in pos}, **{n:0 for n in neg}}
                break
        if mapping:
            y2 = y.map(lambda v: mapping.get(_norm(v), np.nan))
            return y2
        # already numeric/bool? bırak
        return y

    @staticmethod
    def _make_ohe():
        """sklearn sürüm uyumluluğu: sparse_output yoksa sparse kullan."""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    # ---------------------------------------------------------

    def _sense(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        info["n_samples"] = len(X); info["n_features"] = X.shape[1]
        self.task_is_classification = self._is_classification(y)
        info["task"] = "classification" if self.task_is_classification else "regression"
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        info["n_numeric"] = len(self.numeric_cols); info["n_categorical"] = len(self.categorical_cols)
        if self.task_is_classification:
            vc = y.value_counts(normalize=True)
            if len(vc)>0:
                info["min_class_ratio"] = float(vc.min())
                info["imbalanced"] = info["min_class_ratio"] < self.cfg.imbalance_threshold
            else:
                info["min_class_ratio"] = np.nan; info["imbalanced"] = False
        else:
            info["y_skew"] = float(pd.Series(y).skew())
        # leakage quick check
        try:
            numeric = X[self.numeric_cols]
            if not numeric.empty:
                corr = numeric.corrwith(y.astype(float) if self.task_is_classification else y).abs()
                info["max_abs_corr"] = float(corr.max())
                info["leakage_suspect"] = bool(corr.max() > (0.98 if self.task_is_classification else 0.999))
            else:
                info["max_abs_corr"] = np.nan; info["leakage_suspect"] = False
        except Exception:
            info["max_abs_corr"] = np.nan; info["leakage_suspect"] = False
        return info

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        unique = pd.unique(y.dropna())
        return (pd.api.types.is_integer_dtype(y) and len(unique) <= 20) or pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y)

    def _scorer_name(self) -> str:
        return self.cfg.target_metric if self.cfg.target_metric else ("f1_macro" if self.task_is_classification else "r2")

    def _cv(self):
        return StratifiedKFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state) if self.task_is_classification else KFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state)

    def _plan(self, sense_info: Dict[str, Any], prev_results: List[TrialResult]) -> TrialPlan:
        n, p = sense_info["n_samples"], sense_info["n_features"]
        imbalanced = sense_info.get("imbalanced", False)
        p_over_n = p / max(1, n)
        if p_over_n > 1.5:         family = "l1"
        elif sense_info["n_categorical"] > sense_info["n_numeric"]: family = "mi"
        elif imbalanced and self.task_is_classification:            family = "tree"
        else:                                                         family = "kbest"
        params = {"k": min(max(5, p // 4), max(1, p - 1)), "step": 0.2, "C": 1.0, "alpha": 0.001, "n_estimators": 300}
        return TrialPlan(strategy=family, params=params, comment=f"family={family}; p/n={p_over_n:.2f}; imbalanced={imbalanced}")

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())
                ]),
                self.numeric_cols
            ))
        if self.categorical_cols:
            transformers.append((
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", self._make_ohe())
                ]),
                self.categorical_cols
            ))
        if not transformers:
            return ColumnTransformer([], remainder="passthrough")
        return ColumnTransformer(transformers, remainder="drop")

    def _make_selector(self, plan: TrialPlan, task_is_cls: bool, n_features_after_prep: Optional[int] = None) -> Tuple[str, Any]:
        k = plan.params.get("k", 10)
        n_features = n_features_after_prep

        if plan.strategy == "kbest":
            score_fn = f_classif if task_is_cls else f_regression
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"
            links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html" if task_is_cls else "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html"
            if n_features is not None and isinstance(k, int) and k > n_features: k = n_features
            return "sel", SafeSelectKBest(score_func=score_fn, k=k)

        if plan.strategy == "mi":
            score_fn = mutual_info_classif if task_is_cls else mutual_info_regression
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"
            links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html" if task_is_cls else "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html"
            if n_features is not None and isinstance(k, int) and k > n_features: k = n_features
            return "sel", SafeSelectKBest(score_func=score_fn, k=k)

        if plan.strategy == "rfe":
            base = LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if task_is_cls else LinearRegression()
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html"
            links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" if isinstance(base, LogisticRegression) else "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
            return "sel", RFE(estimator=base, n_features_to_select=min(k, n_features) if n_features else k, step=plan.params.get("step", 0.2))

    def _default_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if self.task_is_classification else LinearRegression()

    def _act_build_pipeline(self, plan: TrialPlan, X: Optional[pd.DataFrame] = None) -> Pipeline:
        pre = self._build_preprocessor()
        n_after = None
        if X is not None:
            try:
                Xt = pre.fit_transform(X)
                n_after = Xt.shape[1]
            except Exception:
                n_after = None
        sel_name, selector = self._make_selector(plan, self.task_is_classification, n_features_after_prep=n_after)
        model = self._default_estimator()
        pipe = Pipeline([("prep", pre), (sel_name, selector), ("model", model)])
        return pipe

    def _evaluate(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> TrialResult:
        metric = self._scorer_name(); cv = self._cv()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric, n_jobs=-1, error_score=np.nan)
        start_fit = time.time(); pipe.fit(X, y)
        n_features = self._infer_selected_feature_count(pipe, X)
        selected = self._infer_selected_feature_names(pipe, X)
        duration = (time.time() - start_fit) + 0.0
        return TrialResult(metric_name=metric,
                           metric_value=float(np.nanmean(scores)),
                           metric_std=float(np.nanstd(scores)),
                           n_features=n_features,
                           selected_features=selected,
                           pipeline_repr=str(pipe),
                           duration_sec=duration)

    def _infer_selected_feature_count(self, pipe: Pipeline, X: pd.DataFrame) -> int:
        try:
            sel = pipe.named_steps.get("sel")
            if hasattr(sel, "get_support"):
                Xt = pipe.named_steps["prep"].fit_transform(X)
                return int(sel.fit(Xt, np.zeros(len(X))).get_support().sum())
        except Exception: pass
        return X.shape[1]

    def _infer_selected_feature_names(self, pipe: Pipeline, X: pd.DataFrame) -> List[str]:
        try:
            prep: ColumnTransformer = pipe.named_steps["prep"]; sel = pipe.named_steps.get("sel")
            feature_names = []
            for name, trans, cols in prep.transformers_:
                if name == "remainder": continue
                outs = trans.get_feature_names_out(cols) if hasattr(trans, "get_feature_names_out") else cols
                feature_names.extend(list(outs))
            if hasattr(sel, "get_support"):
                Xt = prep.transform(X); sel.fit(Xt, np.zeros(len(X)))
                mask = sel.get_support()
                if mask is not None and len(feature_names) == len(mask):
                    return [f for f, m in zip(feature_names, mask) if m]
        except Exception: pass
        return list(X.columns)

    def _reflect(self, plan: TrialPlan, result: TrialResult, sense_info: Dict[str, Any]) -> TrialPlan:
        if len(self.history) >= 2 and result.metric_value < self.best_score + 1e-4:
            if plan.strategy in {"kbest", "mi"}:
                plan.params["k"] = max(5, int(plan.params.get("k", 10) * 1.5))
                plan.comment += "; reflect: increase k"
            elif plan.strategy in {"l1", "tree"}:
                plan.strategy = "rfe" if plan.strategy == "l1" else "kbest"
                plan.comment += "; reflect: switch family"
        if sense_info.get("imbalanced", False) and self.task_is_classification and self._scorer_name() not in {"roc_auc","average_precision"}:
            self.cfg.target_metric = "roc_auc"; plan.comment += "; reflect: set metric=roc_auc"
        if self.pubmed_searcher and len(result.selected_features) <= 5:
            try:
                lit_scores = []
                for feature in result.selected_features[:5]:
                    if feature not in self.literature_cache:
                        lit_result = self.pubmed_searcher.search_simple(feature, disease_context=self.disease_context, max_results=3)
                        self.literature_cache[feature] = lit_result
                    lit_scores.append(self.literature_cache[feature]['evidence_score'])
                if lit_scores:
                    avg_lit = sum(lit_scores)/len(lit_scores)
                    if avg_lit < 1.5 and plan.strategy == "kbest":
                        plan.params["k"] = max(5, int(plan.params["k"] * 0.8))
                        plan.comment += f"; lit_adj: reduce k (low_lit={avg_lit:.1f})"
                    elif avg_lit > 3.0:
                        plan.comment += f"; lit_adj: keep strategy (high_lit={avg_lit:.1f})"
            except Exception:
                plan.comment += "; lit_adj: error"
        return plan

    def _stop_check(self, start_time: float, trials_done: int, last_result: Optional[TrialResult]) -> bool:
        if trials_done >= self.cfg.budget_trials: return True
        if self.cfg.budget_seconds is not None and (time.time() - start_time) >= self.cfg.budget_seconds: return True
        if self.cfg.target_threshold is not None and last_result is not None and last_result.metric_value >= self.cfg.target_threshold: return True
        return False

    def _maybe_optuna_tune(self, plan: TrialPlan, X: pd.DataFrame, y: pd.Series) -> TrialPlan:
        if not self.cfg.enable_optuna: return plan
        try:
            import optuna  # noqa: F401
        except Exception:
            return plan
        return plan  # placeholder

    def run(self, X: pd.DataFrame, y: pd.Series, progress_callback=None) -> Dict[str, Any]:
        """Ana akış: y'yi temizle, planla, dene, yansıt."""
        # --- hedefte NaN/etiket düzeltme ---
        y0 = y.copy()
        if y0.isna().any():
            keep = ~y0.isna()
            X, y0 = X.loc[keep].reset_index(drop=True), y0.loc[keep].reset_index(drop=True)
        y0 = self._clean_target(y0)

        start = time.time()
        sense_info = self._sense(X, y0)
        plan = self._plan(sense_info, [])

        trials = 0; last_result: Optional[TrialResult] = None
        while True:
            plan = self._maybe_optuna_tune(plan, X, y0)
            pipe = self._act_build_pipeline(plan, X)
            result = self._evaluate(pipe, X, y0)

            result.selected_features = self.hitl.approve_features(result.selected_features)
            self.store.log_trial(plan, result); self.history.append((plan, result))

            if result.metric_value > self.best_score:
                self.best_score = result.metric_value; self.best_pipeline = pipe; self.best_features = result.selected_features
                self.store.save_artifact("best", {
                    "metric": result.metric_name, "score": result.metric_value,
                    "n_features": result.n_features, "features": result.selected_features,
                    "pipeline": result.pipeline_repr, "plan": asdict(plan),
                })

            trials += 1; last_result = result
            if progress_callback: progress_callback(trials, self.cfg.budget_trials, result)
            plan = self._reflect(plan, result, sense_info)
            if self._stop_check(start, trials, last_result): break

        elapsed = time.time() - start
        return {
            "best_score": self.best_score,
            "best_metric": self._scorer_name(),
            "best_features": self.best_features,
            "trials": trials,
            "elapsed_sec": elapsed,
            "sense_info": sense_info,
            "history_df": self.store.dataframe(),
            "documentation_link": links,
            "literature_cache": self.literature_cache if self.pubmed_searcher else {}
        }
