"""Create a transparent register of researcher-defined market proxy baskets.

This prevents exploratory price portfolios from being mistaken for
GeoSignal-generated or empirically validated sector/security predictions.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "datasets" / "case_exposure_proxy_register.csv"
ROWS = [
    ("redsea_20240112", "Red Sea / Houthi shipping disruption", "Ports and logistics", "ADANIPORTS.NS;CONCOR.NS;GESHIP.NS;SCI.NS;AEGISLOG.NS", "candidate market proxy basket", "Researcher-selected from event theme; firm-level route/cargo exposure not checked", "hypothesis only", False),
    ("redsea_20240112", "Red Sea / Houthi shipping disruption", "Energy", "RELIANCE.NS;ONGC.NS;IOC.NS;BPCL.NS", "candidate market proxy basket", "Researcher-selected for energy price context; physical supply exposure not checked", "hypothesis only", False),
    ("redsea_20240112", "Red Sea / Houthi shipping disruption", "FMCG", "HINDUNILVR.NS;ITC.NS;NESTLEIND.NS;BRITANNIA.NS", "exploratory comparison basket", "Broad consumer-sector comparison; event-specific exposure not checked", "comparison only", False),
    ("chip_controls_20240312", "US assessment of expanded chip export controls affecting China's military", "Global semiconductor firms", "NVDA;TSM;ASML;INTC", "candidate market proxy basket", "Researcher-selected sector proxies; company-specific product/customer exposure not checked", "hypothesis only", False),
    ("chip_controls_20240312", "US assessment of expanded chip export controls affecting China's military", "US semiconductor ETFs", "SOXX;SMH", "sector market context", "Broad sector instruments; not firm-level exposure labels", "context only", False),
    ("chip_controls_20240312", "US assessment of expanded chip export controls affecting China's military", "Indian large-cap IT", "TCS.NS;INFY.NS;HCLTECH.NS;WIPRO.NS", "indirect comparison basket", "No direct semiconductor exposure evidence established", "comparison only", False),
]

def main():
    columns = ["case_id", "event_description", "candidate_sector", "market_series", "portfolio_role", "basis_and_limit", "mapping_status", "validated_exposure"]
    pd.DataFrame(ROWS, columns=columns).to_csv(OUT, index=False)
    print(f"Wrote {len(ROWS)} explicitly unvalidated exposure/proxy mappings to {OUT.name}")

if __name__ == "__main__": main()
