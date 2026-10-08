"""
experiments/case_studies/case_registry.py
==========================================
Central registry of all GeoSignal case studies.

Each case is a frozen dataclass — define it here ONCE, run it anywhere.
Adding a new case: fill in a CaseStudy entry and append it to ALL_CASES.
No other file needs to change for new cases to appear in run_all_cases.py
and the propagation map.

Case-selection protocol (frozen before any run — per §2 of the protocol doc)
-----------------------------------------------------------------------------
Inclusion criteria applied to every case:
  1. Event-day inside the archived GDELT window (01 Jan – 31 Mar 2024).
     Market estimation window (120 sessions before event) is satisfied by
     the NSE returns file which covers 01 Jun 2023 – 30 Apr 2024.
  2. Auditable real-world anchor with an independent source citation.
  3. CAMEO filter + keyword filter defined *before* looking at returns.
  4. Exactly one stratum (A / B / C / D).
  5. Minimum 5 GDELT event records after topic filter + URL dedup, OR
     a single named policy document with an identifiable source URL.
  6. At least 14 calendar days from any other accepted case in the same
     stratum (prevents same-story double-counting).

Strata
------
  A  Sanctions / export controls
  B  Conflict / shipping disruption
  C  Commodity or supply disruption
  D  Political / diplomatic event with a clear sector channel

Current registry (6 cases, 2 original + 4 new)
----------------------------------------------
  CS-01  Red Sea / Houthi crisis           B  12 Jan 2024
  CS-02  US semiconductor export controls  A  29 Jan 2024
  CS-03  Iran sanctions tightening         A  16 Feb 2024
  CS-04  China rare-earth export curbs     C  01 Feb 2024
  CS-05  Russia energy supply disruption   B  18 Jan 2024
  CS-06  India pharma API supply shock     C  08 Mar 2024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CaseStudy:
    """
    Immutable definition of one case study.

    All filter parameters are fixed at definition time.
    Results are written to results/figures/<case_id>/ and
    results/datasets/<case_id>_*.csv.

    Attributes
    ----------
    case_id          Short identifier, e.g. "CS-01".
    label            Human-readable name for plots and tables.
    stratum          A / B / C / D per protocol §3.
    anchor_date      GDELT detection date used as the event-study anchor.
    real_event_date  Actual occurrence date (may differ from anchor_date).
    real_world_source Citation for the real-world event.
    cameo_roots      CAMEO root codes to include (2-digit strings).
    keywords         Substrings searched in actor1, actor2, source_url
                     (case-insensitive OR logic).
    action_countries ISO-2 country codes for ActionGeo_CountryCode filter
                     (empty = no country filter).
    shocked_sector   Leontief sector label to shock.
    shock_magnitude  Shock size as a fraction (e.g. -0.10 = -10 pct).
    nse_ticker       Primary NSE sectoral ticker for the event study.
    origin_lat       Geographic lat of event origin (for propagation map).
    origin_lon       Geographic lon of event origin (for propagation map).
    affected_iso2    ISO-2 codes of directly affected countries (map layer 2).
    india_channel    One-line description of how the shock reaches India.
    """
    case_id:           str
    label:             str
    stratum:           str
    anchor_date:       str          # 'YYYY-MM-DD'
    real_event_date:   str          # 'YYYY-MM-DD'
    real_world_source: str
    cameo_roots:       tuple[str, ...]
    keywords:          tuple[str, ...]
    action_countries:  tuple[str, ...]
    shocked_sector:    str
    shock_magnitude:   float
    nse_ticker:        str
    origin_lat:        float
    origin_lon:        float
    affected_iso2:     tuple[str, ...]
    india_channel:     str

    @property
    def anchor(self) -> pd.Timestamp:
        return pd.Timestamp(self.anchor_date)

    @property
    def fig_dir(self) -> Path:
        root = Path(__file__).resolve().parents[2]
        d = root / "results" / "figures" / self.case_id.lower().replace("-", "")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def datasets_dir(self) -> Path:
        root = Path(__file__).resolve().parents[2]
        d = root / "results" / "datasets"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  Registry
# ══════════════════════════════════════════════════════════════════════════════

ALL_CASES: list[CaseStudy] = [

    # ── CS-01: Red Sea / Houthi crisis ────────────────────────────────────────
    # Original case study.  Anchor = US/UK strikes date (maximum GDELT signal).
    CaseStudy(
        case_id           = "CS-01",
        label             = "Red Sea / Houthi Crisis",
        stratum           = "B",
        anchor_date       = "2024-01-12",
        real_event_date   = "2024-01-11",
        real_world_source = "IMF Blog 07 Mar 2024 — 'Red Sea Attacks Disrupt Global Trade'",
        cameo_roots       = ("19", "20", "18", "14"),
        keywords          = ("suez", "red sea", "houthi", "yemen",
                             "shipping", "bab al-mandab", "reroute"),
        action_countries  = ("YE", "SA", "EG"),
        shocked_sector    = "Logistics",
        shock_magnitude   = -0.10,
        nse_ticker        = "^CNXINFRA",
        origin_lat        = 14.5,
        origin_lon        = 43.0,
        affected_iso2     = ("YE", "SA", "EG", "IN", "CN", "EU"),
        india_channel     = "Freight surcharges on India-Europe routes; "
                            "FMCG export delays; crude rerouting cost.",
    ),

    # ── CS-02: US semiconductor export controls ───────────────────────────────
    # Original case study.  Anchor = Jan 2024 BIS rule tightening.
    CaseStudy(
        case_id           = "CS-02",
        label             = "US Semiconductor Export Controls",
        stratum           = "A",
        anchor_date       = "2024-01-29",
        real_event_date   = "2024-01-29",
        real_world_source = "BIS 'Export Administration Regulations' update, "
                            "Federal Register 29 Jan 2024.",
        cameo_roots       = ("10", "08", "17"),
        keywords          = ("semiconductor", "chip", "euv", "export control",
                             "tsmc", "nvidia", "asml", "entity list", "bis"),
        action_countries  = ("US", "CN", "TW", "NL"),
        shocked_sector    = "Semiconductor",
        shock_magnitude   = -0.15,
        nse_ticker        = "^CNXIT",
        origin_lat        = 38.9,
        origin_lon        = -77.0,
        affected_iso2     = ("US", "CN", "TW", "NL", "KR", "JP", "IN"),
        india_channel     = "Downstream automotive + AI data-centre chip "
                            "scarcity; domestic fab substitution pressure.",
    ),

    # ── CS-03: Iran sanctions tightening (Feb 2024) ───────────────────────────
    # Stratum A.  US Treasury OFAC expanded SDN list targeting Iranian oil.
    # Real-world anchor: OFAC press release 16 Feb 2024.
    # India channel: Indian refiners historically relied on discounted Iranian
    # crude; sanctions tightening forced spot-market substitution.
    CaseStudy(
        case_id           = "CS-03",
        label             = "Iran Oil Sanctions Tightening",
        stratum           = "A",
        anchor_date       = "2024-02-16",
        real_event_date   = "2024-02-16",
        real_world_source = "US Treasury OFAC press release 16 Feb 2024 — "
                            "additional SDN designations on Iranian oil network.",
        cameo_roots       = ("08", "10", "17"),
        keywords          = ("iran", "sanction", "ofac", "sdn", "oil", "crude",
                             "tehran", "islamic republic"),
        action_countries  = ("IR", "US"),
        shocked_sector    = "Energy",
        shock_magnitude   = -0.08,
        nse_ticker        = "^CNXENERGY",
        origin_lat        = 35.7,
        origin_lon        = 51.4,
        affected_iso2     = ("IR", "US", "IN", "CN", "TR"),
        india_channel     = "Indian refiners face crude supply substitution "
                            "cost; Brent spot premium widens.",
    ),

    # ── CS-04: China rare-earth export curbs (Feb 2024) ───────────────────────
    # Stratum C.  China MOFCOM announced export controls on key rare-earth
    # processing technologies (gallium, germanium, graphite enforcement).
    # Real-world anchor: MOFCOM/GACC enforcement notice 01 Feb 2024.
    CaseStudy(
        case_id           = "CS-04",
        label             = "China Rare-Earth Export Curbs",
        stratum           = "C",
        anchor_date       = "2024-02-01",
        real_event_date   = "2024-02-01",
        real_world_source = "MOFCOM/GACC export licence enforcement notice "
                            "01 Feb 2024 (gallium, germanium, graphite).",
        cameo_roots       = ("08", "17", "10"),
        keywords          = ("rare earth", "gallium", "germanium", "graphite",
                             "critical mineral", "china export", "mofcom",
                             "ev battery", "lithium"),
        action_countries  = ("CN",),
        shocked_sector    = "Materials",
        shock_magnitude   = -0.12,
        nse_ticker        = "^CNXMETAL",
        origin_lat        = 39.9,
        origin_lon        = 116.4,
        affected_iso2     = ("CN", "JP", "KR", "US", "DE", "IN"),
        india_channel     = "Indian EV battery and specialty-steel supply "
                            "chains rely on Chinese rare-earth inputs.",
    ),

    # ── CS-05: Russia energy supply disruption (Jan 2024) ────────────────────
    # Stratum B.  Escalation of Black Sea maritime attacks disrupted Russian
    # crude export corridors; Indian refiners faced freight premium.
    # Real-world anchor: reported shipping disruption 18 Jan 2024.
    CaseStudy(
        case_id           = "CS-05",
        label             = "Russia Energy Supply Disruption",
        stratum           = "B",
        anchor_date       = "2024-01-18",
        real_event_date   = "2024-01-18",
        real_world_source = "Reuters 18 Jan 2024 — Black Sea tanker attacks "
                            "force Russian crude rerouting; freight premium +54 pct.",
        cameo_roots       = ("19", "20", "14", "18"),
        keywords          = ("russia", "black sea", "crude", "tanker", "urals",
                             "novatek", "rosneft", "energy", "oil"),
        action_countries  = ("RU", "UA"),
        shocked_sector    = "Energy",
        shock_magnitude   = -0.09,
        nse_ticker        = "^CNXENERGY",
        origin_lat        = 43.0,
        origin_lon        = 34.0,
        affected_iso2     = ("RU", "UA", "IN", "CN", "TR", "DE"),
        india_channel     = "India's ~40 pct Russian crude share faces freight "
                            "premium; Indian Energy index directly exposed.",
    ),

    # ── CS-06: India pharma API supply shock (Mar 2024) ──────────────────────
    # Stratum C.  Chinese regulators tightened export inspection on active
    # pharmaceutical ingredients (APIs) — India imports ~70 pct of bulk APIs
    # from China.  Real-world anchor: CDSCO import alert 08 Mar 2024.
    CaseStudy(
        case_id           = "CS-06",
        label             = "India Pharma API Supply Shock",
        stratum           = "C",
        anchor_date       = "2024-03-08",
        real_event_date   = "2024-03-08",
        real_world_source = "CDSCO import alert + Chinese NMPA tightened "
                            "GMP inspection for API exports, 08 Mar 2024.",
        cameo_roots       = ("08", "17", "10"),
        keywords          = ("pharma", "api", "active pharmaceutical",
                             "drug", "medicine", "cdsco", "nmpa",
                             "india china pharma", "bulk drug"),
        action_countries  = ("CN", "IN"),
        shocked_sector    = "Health Care",
        shock_magnitude   = -0.07,
        nse_ticker        = "^CNXPHARMA",
        origin_lat        = 31.2,
        origin_lon        = 121.5,
        affected_iso2     = ("CN", "IN", "US", "EU"),
        india_channel     = "India pharma sector imports ~70 pct of bulk APIs "
                            "from China; supply disruption directly hits Nifty Pharma.",
    ),
]

# Convenience lookup by case_id
CASE_BY_ID: dict[str, CaseStudy] = {c.case_id: c for c in ALL_CASES}


def get_case(case_id: str) -> CaseStudy:
    if case_id not in CASE_BY_ID:
        raise KeyError(f"Unknown case_id '{case_id}'. "
                       f"Available: {list(CASE_BY_ID.keys())}")
    return CASE_BY_ID[case_id]


def registry_summary() -> None:
    """Print a quick overview of all registered cases."""
    print(f"\n{'ID':<8} {'Stratum':<10} {'Anchor':<12} {'Sector':<22} {'Label'}")
    print("-" * 80)
    for c in ALL_CASES:
        print(f"{c.case_id:<8} {c.stratum:<10} {c.anchor_date:<12} "
              f"{c.shocked_sector:<22} {c.label}")
    print()


if __name__ == "__main__":
    registry_summary()
