"""
Data-derived summary facts, shared by the API (/api/insights) and the
Streamlit dashboard so both always state the same numbers.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
PERCENT = 100
MIN_TYPE_COUNT = 100
TOP_DEADLIEST = 3


def by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents and severe accidents per accident type, most common first."""
    typed = df.assign(type=df["accident_type"].astype("string").str.title())
    g = typed.groupby("type")["incident_severity"].agg(total="size", high="sum").reset_index()
    return g.assign(low=g["total"] - g["high"]).sort_values("total", ascending=False)


def compute_insights(df: pd.DataFrame, shap_importance: Optional[pd.DataFrame]) -> dict[str, Any]:
    season = df.groupby("season")["incident_severity"].mean()
    annual = df.groupby("year").size()
    last_year = int(annual.index.max())
    prov_share = df["province"].value_counts(normalize=True)
    types = by_type(df).assign(rate=lambda g: g["high"] / g["total"])
    deadliest = types[types["total"] >= MIN_TYPE_COUNT].sort_values("rate", ascending=False).head(TOP_DEADLIEST)
    return {
        "severe_rate_pct": round(float(df["incident_severity"].mean()) * PERCENT, 1),
        "severe_rate_by_season_pct": (season * PERCENT).round(1).reindex(SEASON_ORDER).to_dict(),
        "peak_year": int(annual.idxmax()),
        "peak_count": int(annual.max()),
        "last_year": last_year,
        "last_count": int(annual.loc[last_year]),
        "change_peak_to_last_pct": round(float(annual.loc[last_year] / annual.max() - 1) * PERCENT, 1),
        "top_provinces": [{"province": p, "share_pct": round(float(s) * PERCENT, 1)}
                          for p, s in prov_share.head(2).items()],
        "most_common_type": {"type": types.iloc[0]["type"],
                             "share_pct": round(float(types.iloc[0]["total"]) / len(df) * PERCENT, 1)},
        "deadliest_types": [{"type": r.type, "severe_rate_pct": round(float(r.rate) * PERCENT, 1),
                             "total": int(r.total)} for r in deadliest.itertuples()],
        "top_shap_features": shap_importance["feature"].head(3).tolist() if shap_importance is not None else [],
    }
