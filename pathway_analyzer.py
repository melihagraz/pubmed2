# pathway_analyzer.py - Real API-based pathway analysis
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Iterable, Tuple, Union
import math
import requests  # <-- Bu satır önemli
import time
import json
import logging
from urllib.parse import quote

try:
    import pandas as pd
except Exception:
    pd = None

# ======= (A) Yardımcı istatistik: hipergeometrik & BH-FDR =======

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def _hypergeom_pval(N: int, K: int, n: int, k: int) -> float:
    max_i = min(K, n)
    denom = _log_choose(N, n)
    s = 0.0
    for i in range(k, max_i + 1):
        num = _log_choose(K, i) + _log_choose(N - K, n - i)
        s += math.exp(num - denom)
    return min(1.0, max(0.0, s))

def _bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        j = m - rank + 1
        val = (pvals[i] * m) / j
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj

# ======= (B) Basit, gömülü mini veritabanı (yedek mod için) =======
 
# ======= (C) Sonuç veri sınıfı =======

@dataclass
class PathwayResult:
    pathway_name: str
    database: str
    p_value: float
    p_adj: float
    feature_count: int
    pathway_size: int
    background_size: int
    enrichment_score: float
    features: List[str]
    source: str = "auto"  # gseapy | builtin_ora

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ======= (D) Ana analiz sınıfı =======

class PathwayAnalyzer:
    """
    Gerçek pathway enrichment (varsa gseapy.enrichr) + yedek olarak ORA.
    """

    def __init__(
        self,
        database: str = "KEGG",
        cache_enabled: bool = True,
        custom_db: Optional[Dict[str, Set[str]]] = None,
        gene_mapper: Optional[Dict[str, str]] = None,
        organism: str = "Human",
        libraries: Optional[List[str]] = None,  # gseapy Enrichr kütüphaneleri
        use_gseapy: bool = True,
    ):
        self.database = database
        self.cache_enabled = cache_enabled
        self._cache: Dict[Tuple, List[PathwayResult]] = {}
        self.gene_mapper = gene_mapper or {}
        self.organism = organism
        self.use_gseapy = use_gseapy

        # Enrichr library defaults (insan için güncel)
        self.libraries = libraries or [
            "KEGG_2021_Human",
            "Reactome_2016",
            "GO_Biological_Process_2021",
        ]

        # Yerel veritabanı (yedek ORA için)
        if custom_db is not None:
            self.db = {k: set(v) for k, v in custom_db.items()}
        else:
            self.db = {}
        # gseapy var mı?
        self._gseapy = None
        if self.use_gseapy:
            try:
                import gseapy  # type: ignore
                self._gseapy = gseapy
            except Exception:
                self._gseapy = None

    @staticmethod
    def _normalize(x: str) -> str:
        return (x or "").strip().upper()

    def _map_feature(self, f: str) -> str:
        f2 = self._normalize(f)
        return self.gene_mapper.get(f2, f2)

    # ---------- (D1) gseapy.Enrichr motoru ----------
    def _run_gseapy_enrichr(self, features: List[str]) -> List[PathwayResult]:
        if self._gseapy is None:
            return []

        genes = sorted({self._map_feature(f) for f in features})
        if not genes:
            return []

        out: List[PathwayResult] = []
        for lib in self.libraries:
            try:
                enr = self._gseapy.enrichr(gene_list=genes, gene_sets=[lib], organism=self.organism, cutoff=1.0, outdir=None, no_plot=True)
                # gseapy >= 1.1.0: results=dict of DataFrames; eski sürümlerde enr.results de olabilir
                df = None
                if hasattr(enr, "results") and enr.results is not None:
                    if isinstance(enr.results, dict):
                        # tek library yolladık, o yüzden lib anahtarı olmalı
                        df = enr.results.get(lib)
                    else:
                        df = enr.results
                elif hasattr(enr, "res2d"):
                    df = enr.res2d
                if df is None:
                    continue

                if pd is None:
                    # pandas yoksa DataFrame varsayma — atla
                    continue

                # Enrichr DF alanları: Term, Overlap, P-value, Adjusted P-value, Genes, Combined Score
                for _, row in df.iterrows():
                    term = str(row.get("Term", ""))
                    p = float(row.get("P-value", 1.0))
                    q = float(row.get("Adjusted P-value", p))
                    genes_str = str(row.get("Genes", ""))  # "GENE1;GENE2;..."
                    overl = [self._normalize(t) for t in genes_str.replace(",", ";").split(";") if t.strip()]
                    k = len(overl)

                    # Pathway size/Background size Enrichr'de direkt verilmez; yaklaşık:
                    pathway_size = max(k, int(str(row.get("Overlap", "0/0")).split("/")[-1]) if "Overlap" in row else k)
                    background_size = max(pathway_size, 20000)  # kaba tahmin; görsel amaçlı
                    enrich = float(row.get("Combined Score", (k / max(1, pathway_size)) * 10.0))

                    # database ismini lib'den çıkar
                    if lib.upper().startswith("KEGG"):
                        db_name = "KEGG"
                    elif lib.upper().startswith("REACTOME"):
                        db_name = "Reactome"
                    elif "GO_" in lib.upper():
                        db_name = "GO-BP"
                    else:
                        db_name = lib

                    out.append(
                        PathwayResult(
                            pathway_name=term,
                            database=db_name,
                            p_value=p,
                            p_adj=q,
                            feature_count=k,
                            pathway_size=pathway_size,
                            background_size=background_size,
                            enrichment_score=enrich,
                            features=sorted(overl),
                            source="gseapy",
                        )
                    )
            except Exception:
                # bir kütüphane patlarsa diğerleri denensin
                continue

        # sıralama: q, p, k
        out.sort(key=lambda r: (r.p_adj, r.p_value, -r.feature_count))
        return out

    # ---------- (D2) Yedek: ORA mini-DB ----------
    

    # ---------- (D3) Genel arayüz ----------
    def analyze_pathways(
        self,
        features: List[str],
        background: Optional[Iterable[str]] = None,  # şimdilik gseapy için kullanılmıyor
        max_pathways: int = 20,
        min_feature_count: int = 1,
        max_p_value: float = 0.05,
    ) -> List[Dict]:
        key = None
        if self.cache_enabled:
            key = (
                tuple(sorted(map(self._normalize, features))),
                tuple(sorted(self.libraries)),
                self.database,
                self.organism,
                max_pathways,
                min_feature_count,
                max_p_value,
            )
            if key in self._cache:
                return [r.to_dict() for r in self._cache[key]]

        # Önce gseapy dene (varsa)
        results: List[PathwayResult] = []
        if self._gseapy is not None:
            try:
                res = self._run_gseapy_enrichr(features)
                # p-değeri filtresi ve ilk N
                res = [r for r in res if (r.p_value <= max_p_value or r.p_adj <= max_p_value)]
                results = res[:max_pathways]
            except Exception:
                results = []

        # Sonuç yoksa → ORA yedek
        
        if self.cache_enabled and key is not None:
            self._cache[key] = results

        return [r.to_dict() for r in results]


class PathwayVisualizer:
    """Plot/tablolaştırma için yardımcı sınıf."""

    @staticmethod
    def prepare_results_for_plotting(results: List[Dict]) -> "pd.DataFrame | List[Dict]":
        if pd is None:
            # pandas yoksa list döndür
            def _neglog10(x: float) -> float:
                try:
                    return -math.log10(max(float(x), 1e-300))
                except Exception:
                    return 0.0
            out = []
            for r in results:
                r2 = dict(r)
                r2["neg_log_p"] = _neglog10(r.get("p_value", 1.0))
                r2["neg_log_q"] = _neglog10(r.get("p_adj", 1.0))
                out.append(r2)
            return out

        df = pd.DataFrame(results).copy()
        df["neg_log_p"] = df["p_value"].apply(lambda x: -math.log10(max(float(x), 1e-300)))
        df["neg_log_q"] = df["p_adj"].apply(lambda x: -math.log10(max(float(x), 1e-300)))
        return df

    @staticmethod
    def to_table(df) -> "pd.DataFrame | List[Dict]":
        if pd is None:
            return df
        keep = [c for c in [
            "pathway_name","database","p_value","p_adj","feature_count",
            "pathway_size","background_size","enrichment_score","features","source"
        ] if c in df.columns]
        return df[keep].copy()
