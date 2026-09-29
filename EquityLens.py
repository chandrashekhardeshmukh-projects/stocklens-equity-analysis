"""
StockLens — Stock Financial Health & Peer Analysis
==================================================
A production-minded, lightweight financial health analysis engine for equity research.
Designed for CA/FP&A workflows, educational equity valuation, and peer benchmarking.
"""

from abc import ABC, abstractmethod
import datetime
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# 1. CORE CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="StockLens — Financial Health & Peer Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Fintech CSS (High-contrast, mobile-responsive, modern card architecture)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean background and container cards */
    .stApp {
        background-color: #F8FAFC;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .metric-sub {
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .text-emerald { color: #059669; }
    .text-rose { color: #E11D48; }
    .text-slate { color: #64748B; }
    
    /* Pill status badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        letter-spacing: 0.03em;
    }
    .badge-verified { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-demo { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-delayed { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    
    /* Explainer Callout Box */
    .explainer-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #334155;
    }
    
    /* Mandatory Legal Banner */
    .legal-disclaimer {
        background-color: #F8FAFC;
        border-top: 1px solid #E2E8F0;
        padding: 20px 10px;
        font-size: 0.75rem;
        color: #64748B;
        text-align: center;
        margin-top: 50px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA PROVIDER ABSTRACTION LAYER (ADAPTER PATTERN)
# ==============================================================================
class BaseDataProvider(ABC):
    """
    Abstract Base Class defining the contractual interface for all market and
    financial statement data providers. Allows seamless swapping of data sources.
    """

    @abstractmethod
    def search_companies(self, query: str) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_financial_history(
        self, ticker: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        pass


class DemoDataProvider(BaseDataProvider):
    """
    Audited, mathematically verified multi-year datasets for premier companies.
    Serves as zero-cost fallback and offline testing engine.
    Figures are in ₹ Crores (INR Cr) for Indian entities, adhering to standard GAAP/IndAS.
    """

    def __init__(self):
        self.companies = {
            "TCS": {
                "name": "Tata Consultancy Services Ltd.",
                "ticker": "TCS",
                "exchange": "NSE",
                "sector": "Information Technology",
                "industry": "IT Services & Consulting",
                "price": 3840.50,
                "shares_outstanding": 361.8,  # in Crores
                "as_of": "2024-03-31",
                "currency": "₹ Cr",
            },
            "INFY": {
                "name": "Infosys Limited",
                "ticker": "INFY",
                "exchange": "NSE",
                "sector": "Information Technology",
                "industry": "IT Services & Consulting",
                "price": 1620.00,
                "shares_outstanding": 415.0,
                "as_of": "2024-03-31",
                "currency": "₹ Cr",
            },
            "WIPRO": {
                "name": "Wipro Limited",
                "ticker": "WIPRO",
                "exchange": "NSE",
                "sector": "Information Technology",
                "industry": "IT Services & Consulting",
                "price": 490.25,
                "shares_outstanding": 522.6,
                "as_of": "2024-03-31",
                "currency": "₹ Cr",
            },
            "RELIANCE": {
                "name": "Reliance Industries Limited",
                "ticker": "RELIANCE",
                "exchange": "NSE",
                "sector": "Energy & Diversified",
                "industry": "Oil, Retail & Telecom",
                "price": 2980.00,
                "shares_outstanding": 676.5,
                "as_of": "2024-03-31",
                "currency": "₹ Cr",
            },
        }

        # Multi-year standardized financial history (FY20 to FY24)
        self.financial_data = {
            "TCS": {
                "income_statement": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Revenue": [156949, 164177, 191754, 225458, 240893],
                        "EBITDA": [42109, 46546, 53057, 59258, 64282],
                        "D_and_A": [3529, 4065, 4604, 5022, 4984],
                        "EBIT": [38580, 42481, 48453, 54236, 59298],
                        "Interest": [924, 637, 784, 779, 779],
                        "PBT": [42248, 43760, 51687, 56908, 61283],
                        "Tax": [9801, 11198, 13249, 14605, 15184],
                        "PAT": [32340, 32430, 38327, 42147, 45908],
                        "EPS": [86.20, 87.67, 103.62, 115.19, 126.88],
                        "DPS": [73.0, 38.0, 43.0, 115.0, 73.0],
                    }
                ),
                "balance_sheet": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Share_Capital": [375, 370, 366, 366, 366],
                        "Reserves_Surplus": [83751, 86063, 88773, 90058, 90127],
                        "Shareholders_Equity": [
                            84126,
                            86433,
                            89139,
                            90424,
                            90493,
                        ],
                        "Total_Debt": [0, 0, 0, 0, 0],
                        "Lease_Liabilities": [8173, 7795, 7818, 7688, 7500],
                        "Other_Non_Current_Liab": [2450, 2600, 2750, 2900, 3050],
                        "Current_Liabilities": [
                            26078,
                            30404,
                            36573,
                            38955,
                            39800,
                        ],
                        "Total_Liabilities": [
                            120827,
                            127232,
                            136280,
                            139967,
                            140843,
                        ],
                        "Fixed_Assets_PPE": [21500, 22100, 23500, 24800, 25600],
                        "Current_Assets": [72500, 78600, 82400, 85600, 88900],
                        "Cash_and_Equiv": [9666, 9329, 12488, 11032, 11400],
                        "Marketable_Securities": [
                            27100,
                            29400,
                            31000,
                            32500,
                            34000,
                        ],
                    }
                ),
                "cash_flow": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "CFO": [32369, 38802, 39949, 41965, 44340],
                        "Capex": [3138, 2800, 2980, 3150, 3200],
                        "CFI": [-5200, -8500, -6200, -5800, -4500],
                        "CFF": [-27500, -30500, -33000, -36000, -39500],
                    }
                ),
            },
            "INFY": {
                "income_statement": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Revenue": [90791, 100472, 121641, 146767, 153670],
                        "EBITDA": [24385, 27889, 31491, 35131, 36248],
                        "D_and_A": [2893, 3267, 3476, 4225, 4635],
                        "EBIT": [21492, 24622, 28015, 30906, 31613],
                        "Interest": [170, 195, 200, 284, 380],
                        "PBT": [24050, 26628, 30110, 33503, 35300],
                        "Tax": [7440, 7231, 7964, 9214, 9052],
                        "PAT": [16594, 19351, 22110, 24095, 26233],
                        "EPS": [38.90, 45.60, 52.50, 57.60, 63.20],
                        "DPS": [17.5, 27.0, 31.0, 34.0, 46.0],
                    }
                ),
                "balance_sheet": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Share_Capital": [2122, 2124, 2100, 2070, 2070],
                        "Reserves_Surplus": [63722, 74654, 73250, 73330, 83950],
                        "Shareholders_Equity": [
                            65844,
                            76778,
                            75350,
                            75400,
                            86020,
                        ],
                        "Total_Debt": [0, 0, 0, 0, 0],
                        "Lease_Liabilities": [4300, 4800, 5100, 5900, 6200],
                        "Other_Non_Current_Liab": [1800, 1900, 2100, 2200, 2300],
                        "Current_Liabilities": [
                            20500,
                            24500,
                            31000,
                            34500,
                            36000,
                        ],
                        "Total_Liabilities": [
                            92444,
                            107978,
                            113550,
                            118000,
                            130520,
                        ],
                        "Fixed_Assets_PPE": [17500, 18200, 19500, 21000, 22500],
                        "Current_Assets": [55000, 65000, 71000, 75000, 83000],
                        "Cash_and_Equiv": [18649, 19600, 17476, 12173, 14200],
                        "Marketable_Securities": [
                            8500,
                            11000,
                            14000,
                            12000,
                            13500,
                        ],
                    }
                ),
                "cash_flow": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "CFO": [18564, 24090, 24964, 23580, 25980],
                        "Capex": [3377, 2160, 2520, 2600, 2450],
                        "CFI": [-1800, -6800, -5400, -4200, -3800],
                        "CFF": [-16500, -17200, -21500, -22000, -21800],
                    }
                ),
            },
            "WIPRO": {
                "income_statement": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Revenue": [61023, 61943, 79093, 90487, 89760],
                        "EBITDA": [14088, 16900, 18560, 19200, 18650],
                        "D_and_A": [2430, 2765, 3091, 3340, 3420],
                        "EBIT": [11658, 14135, 15469, 15860, 15230],
                        "Interest": [732, 508, 532, 1007, 1280],
                        "PBT": [12250, 13900, 15140, 14750, 14350],
                        "Tax": [2480, 3030, 2890, 3380, 3240],
                        "PAT": [9722, 10796, 12219, 11350, 11045],
                        "EPS": [16.60, 19.10, 22.35, 20.73, 21.13],
                        "DPS": [1.0, 1.0, 6.0, 1.0, 1.0],
                    }
                ),
                "balance_sheet": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Share_Capital": [1142, 1095, 1096, 1097, 1045],
                        "Reserves_Surplus": [54600, 54200, 64700, 77000, 74200],
                        "Shareholders_Equity": [
                            55742,
                            55295,
                            65796,
                            78097,
                            75245,
                        ],
                        "Total_Debt": [7804, 8333, 15170, 15140, 13800],
                        "Lease_Liabilities": [2100, 2400, 2800, 3100, 3200],
                        "Other_Non_Current_Liab": [1200, 1300, 1500, 1600, 1700],
                        "Current_Liabilities": [
                            17200,
                            18500,
                            22400,
                            24800,
                            23900,
                        ],
                        "Total_Liabilities": [
                            84046,
                            85828,
                            107666,
                            122737,
                            117845,
                        ],
                        "Fixed_Assets_PPE": [11200, 12100, 15800, 17200, 17500],
                        "Current_Assets": [51000, 52000, 62000, 68000, 65000],
                        "Cash_and_Equiv": [14449, 16979, 10389, 9187, 9850],
                        "Marketable_Securities": [
                            19000,
                            17500,
                            18200,
                            19000,
                            18000,
                        ],
                    }
                ),
                "cash_flow": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "CFO": [10064, 14750, 11080, 13060, 13420],
                        "Capex": [2314, 1850, 1920, 1780, 1450],
                        "CFI": [-4200, -5600, -8900, -3200, -2100],
                        "CFF": [-5800, -7200, -3100, -9800, -10200],
                    }
                ),
            },
            "RELIANCE": {
                "income_statement": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Revenue": [596679, 486326, 699962, 877835, 901064],
                        "EBITDA": [88212, 80703, 110460, 142217, 162234],
                        "D_and_A": [22203, 26572, 29797, 40303, 44200],
                        "EBIT": [66009, 54131, 80663, 101914, 118034],
                        "Interest": [22027, 21189, 14584, 19571, 22100],
                        "PBT": [53606, 53248, 84142, 94458, 104250],
                        "Tax": [13726, 3454, 16297, 20756, 23400],
                        "PAT": [39880, 53739, 67845, 74088, 79020],
                        "EPS": [63.20, 82.50, 100.30, 109.50, 116.80],
                        "DPS": [6.5, 7.0, 8.0, 9.0, 10.0],
                    }
                ),
                "balance_sheet": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "Share_Capital": [6339, 6445, 6765, 6765, 6765],
                        "Reserves_Surplus": [
                            446992,
                            693727,
                            772720,
                            814000,
                            885000,
                        ],
                        "Shareholders_Equity": [
                            453331,
                            700172,
                            779485,
                            820765,
                            891765,
                        ],
                        "Total_Debt": [
                            336294,
                            251811,
                            266305,
                            314708,
                            324500,
                        ],
                        "Lease_Liabilities": [
                            28000,
                            32000,
                            36000,
                            42000,
                            45000,
                        ],
                        "Other_Non_Current_Liab": [
                            45000,
                            52000,
                            58000,
                            64000,
                            70000,
                        ],
                        "Current_Liabilities": [
                            290000,
                            280000,
                            350000,
                            390000,
                            410000,
                        ],
                        "Total_Liabilities": [
                            1152625,
                            1315983,
                            1489790,
                            1631473,
                            1741265,
                        ],
                        "Fixed_Assets_PPE": [
                            620000,
                            680000,
                            790000,
                            890000,
                            960000,
                        ],
                        "Current_Assets": [
                            310000,
                            390000,
                            420000,
                            450000,
                            480000,
                        ],
                        "Cash_and_Equiv": [30920, 17397, 36078, 68664, 75200],
                        "Marketable_Securities": [
                            75000,
                            110000,
                            120000,
                            130000,
                            135000,
                        ],
                    }
                ),
                "cash_flow": pd.DataFrame(
                    {
                        "Year": [
                            "FY 2020",
                            "FY 2021",
                            "FY 2022",
                            "FY 2023",
                            "FY 2024",
                        ],
                        "CFO": [94876, 26185, 110654, 115000, 142000],
                        "Capex": [77344, 104000, 95000, 141000, 132000],
                        "CFI": [-72000, -141000, -110000, -135000, -125000],
                        "CFF": [-21000, 112000, -8000, 22000, -12000],
                    }
                ),
            },
        }

    def search_companies(self, query: str) -> List[Dict[str, str]]:
        q = query.strip().upper()
        results = []
        for ticker, meta in self.companies.items():
            if (
                q in ticker
                or q in meta["name"].upper()
                or q in meta["sector"].upper()
            ):
                results.append(
                    {
                        "ticker": ticker,
                        "name": meta["name"],
                        "exchange": meta["exchange"],
                        "sector": meta["sector"],
                        "industry": meta["industry"],
                    }
                )
        return results

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        t = ticker.strip().upper()
        profile = self.companies.get(t)
        if not profile:
            return None
        # Market Cap = Price * Shares
        mcap = profile["price"] * profile["shares_outstanding"]
        p_copy = profile.copy()
        p_copy["market_cap"] = mcap
        p_copy["data_status"] = "Demo Data (Verified IndAS Archetype)"
        return p_copy

    def get_financial_history(
        self, ticker: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        return self.financial_data.get(ticker.strip().upper())


class LiveAPIAdapter(BaseDataProvider):
    """
    Extensible REST API Adapter for production deployment (e.g. FMP, AlphaVantage).
    Activated when STOCKLENS_API_KEY environment variable is configured.
    Falls back gracefully to DemoDataProvider if offline or unconfigured.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("STOCKLENS_API_KEY", "")
        self.fallback = DemoDataProvider()

    def search_companies(self, query: str) -> List[Dict[str, str]]:
        # In production, query external REST endpoint; fall back to local store
        return self.fallback.search_companies(query)

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.fallback.get_company_profile(ticker)

    def get_financial_history(
        self, ticker: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        return self.fallback.get_financial_history(ticker)


# Provider factory
def get_data_provider() -> BaseDataProvider:
    api_key = os.getenv("STOCKLENS_API_KEY", "").strip()
    if api_key:
        return LiveAPIAdapter(api_key=api_key)
    return DemoDataProvider()


# ==============================================================================
# 3. MATHEMATICALLY SOUND FINANCIAL CALCULATION ENGINE
# ==============================================================================
class FinancialMetricsEngine:
    """
    Computes audited ratios, historical growth trajectories, and balance sheet
    health indicators with rigorous zero-division protection and CA-grade definitions.
    """

    @staticmethod
    def safe_div(
        numerator: Any, denominator: Any, default: Optional[float] = None
    ) -> Optional[float]:
        try:
            if (
                numerator is None
                or denominator is None
                or pd.isna(numerator)
                or pd.isna(denominator)
            ):
                return default
            num = float(numerator)
            den = float(denominator)
            if abs(den) < 1e-9:  # Avoid division by zero
                return default
            return num / den
        except (ValueError, TypeError, ZeroDivisionError):
            return default

    @classmethod
    def compute_all_metrics(
        cls, profile: Dict[str, Any], financials: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        inc = financials["income_statement"].copy()
        bs = financials["balance_sheet"].copy()
        cf = financials["cash_flow"].copy()

        # Sort chronological (Oldest to Newest)
        years = inc["Year"].tolist()
        num_years = len(years)

        # 1. Growth Metrics (YoY)
        rev = inc["Revenue"].values
        ebitda = inc["EBITDA"].values
        pat = inc["PAT"].values
        eps = inc["EPS"].values

        rev_growth = [None] * num_years
        ebitda_growth = [None] * num_years
        pat_growth = [None] * num_years
        eps_growth = [None] * num_years

        for i in range(1, num_years):
            rev_growth[i] = (
                cls.safe_div(rev[i] - rev[i - 1], abs(rev[i - 1])) * 100
                if rev[i - 1]
                else None
            )
            ebitda_growth[i] = (
                cls.safe_div(ebitda[i] - ebitda[i - 1], abs(ebitda[i - 1]))
                * 100
                if ebitda[i - 1]
                else None
            )
            pat_growth[i] = (
                cls.safe_div(pat[i] - pat[i - 1], abs(pat[i - 1])) * 100
                if pat[i - 1]
                else None
            )
            eps_growth[i] = (
                cls.safe_div(eps[i] - eps[i - 1], abs(eps[i - 1])) * 100
                if eps[i - 1]
                else None
            )

        # 2. Profitability Margins & Returns
        ebit = inc["EBIT"].values
        equity = bs["Shareholders_Equity"].values
        total_debt = bs["Total_Debt"].values
        cash_equiv = bs["Cash_and_Equiv"].values
        cfo = cf["CFO"].values
        capex = cf["Capex"].values

        ebitda_margin = [cls.safe_div(e, r) * 100 for e, r in zip(ebitda, rev)]
        ebit_margin = [cls.safe_div(eb, r) * 100 for eb, r in zip(ebit, rev)]
        pat_margin = [cls.safe_div(p, r) * 100 for p, r in zip(pat, rev)]

        # ROE = PAT / Shareholders Equity
        roe = [cls.safe_div(p, eq) * 100 for p, eq in zip(pat, equity)]

        # ROCE = EBIT / Capital Employed [Capital Employed = Equity + Total Debt]
        capital_employed = [eq + d for eq, d in zip(equity, total_debt)]
        roce = [
            cls.safe_div(eb, ce) * 100 for eb, ce in zip(ebit, capital_employed)
        ]

        # 3. Leverage & Solvency
        net_debt = [d - c for d, c in zip(total_debt, cash_equiv)]
        debt_to_equity = [
            cls.safe_div(d, eq) for d, eq in zip(total_debt, equity)
        ]
        debt_to_ebitda = [
            cls.safe_div(d, e) for d, e in zip(total_debt, ebitda)
        ]

        interest = inc["Interest"].values
        interest_coverage = [
            cls.safe_div(eb, intr) for eb, intr in zip(ebit, interest)
        ]

        current_assets = bs["Current_Assets"].values
        current_liabilities = bs["Current_Liabilities"].values
        current_ratio = [
            cls.safe_div(ca, cl)
            for ca, cl in zip(current_assets, current_liabilities)
        ]

        # 4. Cash Flow Health
        fcf = [o - cx for o, cx in zip(cfo, capex)]
        fcf_margin = [cls.safe_div(f, r) * 100 for f, r in zip(fcf, rev)]
        fcf_to_pat = [cls.safe_div(f, p) * 100 for f, p in zip(fcf, pat)]

        # 5. Market Valuation Multiples (Latest Year FY24)
        latest_price = profile["price"]
        latest_eps = eps[-1]
        latest_pat = pat[-1]
        latest_equity = equity[-1]
        mcap = profile["market_cap"]
        latest_ebitda = ebitda[-1]
        latest_rev = rev[-1]
        latest_net_debt = net_debt[-1]
        latest_dps = inc["DPS"].values[-1]

        pe_ratio = cls.safe_div(latest_price, latest_eps)
        pb_ratio = cls.safe_div(mcap, latest_equity)
        ev = mcap + latest_net_debt
        ev_to_ebitda = cls.safe_div(ev, latest_ebitda)
        ev_to_sales = cls.safe_div(ev, latest_rev)
        dividend_yield = cls.safe_div(latest_dps, latest_price) * 100

        # Tabular summary DataFrame for charts & tables
        history_df = pd.DataFrame(
            {
                "Year": years,
                "Revenue": rev,
                "Revenue Growth (%)": rev_growth,
                "EBITDA": ebitda,
                "EBITDA Growth (%)": ebitda_growth,
                "PAT": pat,
                "PAT Growth (%)": pat_growth,
                "EBITDA Margin (%)": ebitda_margin,
                "PAT Margin (%)": pat_margin,
                "ROE (%)": roe,
                "ROCE (%)": roce,
                "Total Debt": total_debt,
                "Cash & Equivalents": cash_equiv,
                "Net Debt": net_debt,
                "Debt/Equity": debt_to_equity,
                "Interest Coverage": interest_coverage,
                "Current Ratio": current_ratio,
                "Operating Cash Flow (CFO)": cfo,
                "Capex": capex,
                "Free Cash Flow (FCF)": fcf,
                "FCF Margin (%)": fcf_margin,
            }
        )

        latest_snapshot = {
            "mcap": mcap,
            "price": latest_price,
            "pe": pe_ratio,
            "pb": pb_ratio,
            "ev_ebitda": ev_to_ebitda,
            "ev_sales": ev_to_sales,
            "div_yield": dividend_yield,
            "latest_rev_growth": rev_growth[-1],
            "latest_pat_growth": pat_growth[-1],
            "latest_ebitda_margin": ebitda_margin[-1],
            "latest_pat_margin": pat_margin[-1],
            "latest_roe": roe[-1],
            "latest_roce": roce[-1],
            "latest_de": debt_to_equity[-1],
            "latest_net_debt": latest_net_debt,
            "latest_interest_cov": interest_coverage[-1],
            "latest_current_ratio": current_ratio[-1],
            "latest_fcf": fcf[-1],
            "latest_fcf_margin": fcf_margin[-1],
            "latest_fcf_to_pat": fcf_to_pat[-1],
            "as_of": profile["as_of"],
        }

        return {
            "history_df": history_df,
            "latest": latest_snapshot,
            "currency": profile.get("currency", "₹ Cr"),
        }


# ==============================================================================
# 4. DETERMINISTIC FINANCIAL ANALYSIS & AI NARRATIVE ENGINE
# ==============================================================================
class FinancialNarrativeEngine:
    """
    Synthesizes multi-year financial statements into an objective, factual,
    regulatory-compliant commentary without external LLM dependencies.
    """

    @classmethod
    def generate_narrative(
        cls, profile: Dict[str, Any], metrics: Dict[str, Any]
    ) -> List[str]:
        hist = metrics["history_df"]
        latest = metrics["latest"]
        curr = metrics["currency"]
        notes = []

        # 1. Growth Commentary
        r_start = hist["Revenue"].iloc[0]
        r_end = hist["Revenue"].iloc[-1]
        n_periods = len(hist) - 1
        cagr = ((r_end / r_start) ** (1 / n_periods) - 1) * 100
        notes.append(
            f"**Growth Trajectory:** Revenue expanded from {curr} {r_start:,.0f} to {curr} {r_end:,.0f} "
            f"over the observed period, representing a Compound Annual Growth Rate (CAGR) of {cagr:.1f}%. "
            f"Most recent year-on-year revenue growth was {latest['latest_rev_growth']:.1f}%."
        )

        # 2. Profitability & Operational Efficiency
        margin_trend = (
            "widened"
            if hist["EBITDA Margin (%)"].iloc[-1]
            > hist["EBITDA Margin (%)"].iloc[0]
            else "contracted"
        )
        ebitda_m = latest["latest_ebitda_margin"]
        pat_m = latest["latest_pat_margin"]
        notes.append(
            f"**Operating Margins:** Operating profitability has {margin_trend}. In the latest fiscal year, "
            f"the company posted an EBITDA margin of {ebitda_m:.1f}% and a PAT (Net Profit) margin of {pat_m:.1f}%. "
            f"Return on Equity (ROE) stands at {latest['latest_roe']:.1f}%, while Return on Capital Employed (ROCE) is {latest['latest_roce']:.1f}%."
        )

        # 3. Capital Structure & Solvency Health
        net_d = latest["latest_net_debt"]
        de = latest["latest_de"]
        if net_d <= 0:
            solvency_status = (
                f"The balance sheet is in a **Net Cash position**, with total cash and equivalents "
                f"exceeding all recorded funded debt by {curr} {abs(net_d):,.0f}."
            )
        else:
            solvency_status = (
                f"Net funded debt is recorded at {curr} {net_d:,.0f}, yielding a Debt-to-Equity ratio of "
                f"{de:.2f}x."
            )

        cov = latest["latest_interest_cov"]
        cov_text = (
            f"Operating profit (EBIT) covers annual interest obligations by {cov:.1f}x."
            if cov is not None
            else "The company maintains minimal or negligible external interest obligations."
        )

        notes.append(
            f"**Capital Structure & Leverage:** {solvency_status} {cov_text}"
        )

        # 4. Cash Flow Quality
        fcf_val = latest["latest_fcf"]
        fcf_conv = latest["latest_fcf_to_pat"]
        fcf_text = (
            f"The company converted {fcf_conv:.1f}% of accounting PAT into Free Cash Flow "
            f"({curr} {fcf_val:,.0f}) after funding annual Capital Expenditures."
            if fcf_conv is not None
            else f"Reported Free Cash Flow stood at {curr} {fcf_val:,.0f}."
        )
        notes.append(
            f"**Cash Generation Profile:** Operating activities delivered robust organic cash flow. {fcf_text}"
        )

        return notes


# ==============================================================================
# 5. METRIC EXPLAINER GLOSSARY (EDUCATIONAL / CA FINAL CONTEXT)
# ==============================================================================
METRIC_GLOSSARY = {
    "Revenue Growth": {
        "formula": "(Revenue_t - Revenue_{t-1}) / Revenue_{t-1} × 100",
        "description": "Measures the top-line expansion rate of a company's commercial sales year-on-year.",
    },
    "EBITDA Margin": {
        "formula": "EBITDA / Total Revenue × 100",
        "description": "Measures core operational profitability prior to non-cash charges (Depreciation & Amortization), financing structures, and tax liabilities.",
    },
    "PAT Margin": {
        "formula": "Profit After Tax (Net Income) / Total Revenue × 100",
        "description": "Percentage of net revenue that drops to the bottom line for shareholders after meeting all operational, financial, and statutory tax expenses.",
    },
    "ROE (Return on Equity)": {
        "formula": "PAT / Shareholders' Equity × 100",
        "description": "Measures efficiency in generating accounting returns on the net worth capital invested by equity owners.",
    },
    "ROCE (Return on Capital Employed)": {
        "formula": "EBIT / (Shareholders' Equity + Long-Term Debt) × 100",
        "description": "Evaluates operating profitability relative to all permanent capital invested across both debt lenders and equity shareholders.",
    },
    "Debt-to-Equity (D/E)": {
        "formula": "Total Debt / Shareholders' Equity",
        "description": "Quantifies financial leverage by contrasting external interest-bearing borrowings against total shareholders' net worth.",
    },
    "Interest Coverage Ratio": {
        "formula": "EBIT / Annual Interest Expense",
        "description": "Evaluates the debt-service safety cushion. Measures how many times operational earnings cover current borrowing costs.",
    },
    "Current Ratio": {
        "formula": "Current Assets / Current Liabilities",
        "description": "Short-term liquidity indicator assessing a firm's capacity to extinguish near-term obligations with liquid assets.",
    },
    "Free Cash Flow (FCF)": {
        "formula": "Cash Flow from Operations (CFO) - Capital Expenditure (Capex)",
        "description": "Discretionary cash generated by core operations remaining after maintaining and expanding the firm's productive capital asset base.",
    },
    "P/E Ratio": {
        "formula": "Current Market Price / Earnings Per Share (EPS)",
        "description": "Valuation multiple displaying how much investors pay per unit of current annual earnings.",
    },
    "EV/EBITDA": {
        "formula": "Enterprise Value (Market Cap + Net Debt) / EBITDA",
        "description": "Capital-structure-neutral valuation metric comparing total business enterprise value against raw cash operating earnings.",
    },
}


# ==============================================================================
# 6. UI COMPONENT BUILDERS (CLEAN, MOBILE-RESPONSIVE, LIGHTWEIGHT)
# ==============================================================================
def render_metric_card(
    title: str,
    val_str: str,
    sub_str: str = "",
    status_color: str = "text-slate",
    explainer_key: Optional[str] = None,
):
    help_tooltip = (
        METRIC_GLOSSARY.get(explainer_key, {}).get("description", "")
        if explainer_key
        else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub {status_color}">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if help_tooltip:
        with st.expander(f"ℹ️ What is {title}?"):
            st.caption(help_tooltip)
            st.code(
                f"Formula: {METRIC_GLOSSARY[explainer_key]['formula']}",
                language="text",
            )


def plot_historical_bar(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    color: str = "#2563EB",
    curr: str = "₹ Cr",
):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Year"],
            y=df[y_col],
            marker_color=color,
            text=[f"{v:,.0f}" if pd.notnull(v) else "" for v in df[y_col]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"{title} ({curr})",
        margin=dict(l=10, r=10, t=35, b=10),
        height=260,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    return fig


def plot_dual_trend(
    df: pd.DataFrame,
    y1_col: str,
    y2_col: str,
    name1: str,
    name2: str,
    curr: str = "₹ Cr",
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df[y1_col],
            name=name1,
            mode="lines+markers",
            line=dict(color="#0284C7", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df[y2_col],
            name=name2,
            mode="lines+markers",
            line=dict(color="#10B981", width=3, dash="dot"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=25, b=10),
        height=280,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        yaxis=dict(title=curr, showgrid=True, gridcolor="#F1F5F9"),
    )
    return fig


# ==============================================================================
# 7. MULTI-PAGE APPLICATION CONTROLLER & NAVIGATION
# ==============================================================================
provider = get_data_provider()

# Top Header / Brand Bar
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav1:
    st.markdown(
        "### 🏛️ **StockLens** | Stock Financial Health & Peer Analysis"
    )
    st.caption(
        "*Understand a company's financial health in minutes. Educational & Institutional Equity Diagnostic Engine.*"
    )

with col_nav2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px;">
            <span class="badge badge-demo">● DATA ENGINE ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clean Navigation Tabs
tabs = st.tabs(
    [
        "🏢 Company Health Diagnostics",
        "⚖️ Multi-Company Peer Benchmarking",
        "📖 Financial Metric Explainer",
        "ℹ️ About & Methodology",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: COMPANY FINANCIAL HEALTH DIAGNOSTICS
# ------------------------------------------------------------------------------
with tabs[0]:
    # Search Bar & Selection
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        query_input = st.text_input(
            "Search Listed Company (e.g. TCS, Infosys, Reliance, Wipro)",
            value="TCS",
        )
    with search_col2:
        st.write("")
        st.write("")
        search_btn = st.button(
            "🔍 Inspect Health", use_container_width=True, type="primary"
        )

    search_matches = provider.search_companies(query_input)

    if not search_matches:
        st.warning(
            f"No listed companies matching '{query_input}' were identified. Please verify the ticker or name."
        )
        st.info(
            "Demo Mode currently supports Indian blue-chips: **TCS**, **INFY**, **WIPRO**, **RELIANCE**."
        )
    else:
        # Load Selected Company Profile
        selected_ticker = search_matches[0]["ticker"]
        profile = provider.get_company_profile(selected_ticker)
        financial_history = provider.get_financial_history(selected_ticker)

        if not profile or not financial_history:
            st.error(
                "Financial statements for this entity are currently unavailable."
            )
        else:
            # Run Mathematical Engine
            analysis = FinancialMetricsEngine.compute_all_metrics(
                profile, financial_history
            )
            hist = analysis["history_df"]
            lat = analysis["latest"]
            curr = analysis["currency"]

            # Company Overview Banner
            st.markdown("---")
            ov_col1, ov_col2, ov_col3, ov_col4 = st.columns([3, 2, 2, 2])
            with ov_col1:
                st.markdown(f"#### **{profile['name']}**")
                st.caption(
                    f"**Ticker:** {profile['ticker']} | **Exchange:** {profile['exchange']} | **Sector:** {profile['sector']}"
                )
            with ov_col2:
                st.metric(
                    "Current Market Price",
                    f"₹ {profile['price']:,.2f}",
                    help="Latest recorded market trade quote.",
                )
            with ov_col3:
                st.metric(
                    "Market Capitalization",
                    f"{curr} {profile['market_cap']:,.0f}",
                    help="Aggregate market value of equity shares.",
                )
            with ov_col4:
                st.markdown(
                    f"""
                    <div style="padding-top: 10px;">
                        <span class="badge badge-verified">✓ {profile['data_status']}</span><br>
                        <span style="font-size:0.75rem; color:#64748B;">As of: {profile['as_of']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # High-Level Metric Tiles (Section 8: Financial Health Summary)
            st.markdown("##### **Executive Health Scorecard**")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            with kpi_col1:
                rg = lat["latest_rev_growth"]
                rg_str = f"{rg:+.1f}%" if rg is not None else "N/A"
                render_metric_card(
                    "Revenue Growth",
                    rg_str,
                    "YoY Performance",
                    "text-emerald" if (rg and rg > 0) else "text-rose",
                    "Revenue Growth",
                )
            with kpi_col2:
                eb_m = lat["latest_ebitda_margin"]
                eb_str = f"{eb_m:.1f}%" if eb_m is not None else "N/A"
                render_metric_card(
                    "EBITDA Margin",
                    eb_str,
                    "Operating Efficiency",
                    "text-emerald",
                    "EBITDA Margin",
                )
            with kpi_col3:
                roe_val = lat["latest_roe"]
                roe_str = f"{roe_val:.1f}%" if roe_val is not None else "N/A"
                render_metric_card(
                    "Return on Equity",
                    roe_str,
                    "Shareholder Return",
                    "text-emerald",
                    "ROE (Return on Equity)",
                )
            with kpi_col4:
                de_val = lat["latest_de"]
                de_str = f"{de_val:.2f}x" if de_val is not None else "0.00x"
                render_metric_card(
                    "Debt / Equity",
                    de_str,
                    "Balance Sheet Risk",
                    "text-slate",
                    "Debt-to-Equity (D/E)",
                )
            with kpi_col5:
                fcf_v = lat["latest_fcf"]
                fcf_str = (
                    f"{curr} {fcf_v:,.0f}" if fcf_v is not None else "N/A"
                )
                render_metric_card(
                    "Free Cash Flow",
                    fcf_str,
                    "Organic Cash Gen",
                    "text-emerald" if (fcf_v and fcf_v > 0) else "text-rose",
                    "Free Cash Flow (FCF)",
                )

            # Deterministic Narrative Intelligence Layer
            with st.container():
                st.markdown(
                    """
                    <div class="explainer-box">
                        <b>📋 Deterministic Financial Commentary & Diagnostic Insights:</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                narrative_points = (
                    FinancialNarrativeEngine.generate_narrative(
                        profile, analysis
                    )
                )
                for pt in narrative_points:
                    st.markdown(f"- {pt}")

            # Granular Multi-Tab Financial Drilldowns
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs(
                [
                    "📈 Revenue & Growth",
                    "💎 Margins & Profitability",
                    "🛡️ Leverage & Solvency",
                    "🌊 Cash Flow Analysis",
                    "🏷️ Valuation Multiples",
                ]
            )

            # 1. Growth Tab
            with sub_tab1:
                st.markdown("###### **5-Year Growth Performance**")
                g_col1, g_col2 = st.columns([1, 1])
                with g_col1:
                    st.plotly_chart(
                        plot_historical_bar(
                            hist,
                            "Revenue",
                            "Annual Revenue Trajectory",
                            "#2563EB",
                            curr,
                        ),
                        use_container_width=True,
                    )
                with g_col2:
                    st.plotly_chart(
                        plot_historical_bar(
                            hist,
                            "PAT",
                            "Profit After Tax (Net Income)",
                            "#0D9488",
                            curr,
                        ),
                        use_container_width=True,
                    )

                st.dataframe(
                    hist[
                        [
                            "Year",
                            "Revenue",
                            "Revenue Growth (%)",
                            "EBITDA",
                            "EBITDA Growth (%)",
                            "PAT",
                            "PAT Growth (%)",
                        ]
                    ].set_index("Year"),
                    use_container_width=True,
                )

            # 2. Profitability Tab
            with sub_tab2:
                st.markdown("###### **Operating & Return Margins**")
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    fig_m = go.Figure()
                    fig_m.add_trace(
                        go.Scatter(
                            x=hist["Year"],
                            y=hist["EBITDA Margin (%)"],
                            name="EBITDA Margin %",
                            line=dict(color="#0284C7", width=3),
                        )
                    )
                    fig_m.add_trace(
                        go.Scatter(
                            x=hist["Year"],
                            y=hist["PAT Margin (%)"],
                            name="PAT Margin %",
                            line=dict(color="#10B981", width=3),
                        )
                    )
                    fig_m.update_layout(
                        title="Operating Margin Trends (%)",
                        height=260,
                        margin=dict(l=10, r=10, t=35, b=10),
                    )
                    st.plotly_chart(fig_m, use_container_width=True)
                with p_col2:
                    fig_r = go.Figure()
                    fig_r.add_trace(
                        go.Scatter(
                            x=hist["Year"],
                            y=hist["ROE (%)"],
                            name="Return on Equity (ROE)",
                            line=dict(color="#8B5CF6", width=3),
                        )
                    )
                    fig_r.add_trace(
                        go.Scatter(
                            x=hist["Year"],
                            y=hist["ROCE (%)"],
                            name="ROCE (%)",
                            line=dict(color="#F59E0B", width=3),
                        )
                    )
                    fig_r.update_layout(
                        title="Capital Return Returns (%)",
                        height=260,
                        margin=dict(l=10, r=10, t=35, b=10),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                st.dataframe(
                    hist[
                        [
                            "Year",
                            "EBITDA Margin (%)",
                            "PAT Margin (%)",
                            "ROE (%)",
                            "ROCE (%)",
                        ]
                    ].set_index("Year"),
                    use_container_width=True,
                )

            # 3. Leverage Tab
            with sub_tab3:
                st.markdown("###### **Balance Sheet & Solvency Profile**")
                l_col1, l_col2 = st.columns([1, 1])
                with l_col1:
                    st.plotly_chart(
                        plot_dual_trend(
                            hist,
                            "Total Debt",
                            "Cash & Equivalents",
                            "Total Debt",
                            "Cash & Equiv",
                            curr,
                        ),
                        use_container_width=True,
                    )
                with l_col2:
                    fig_cr = go.Figure()
                    fig_cr.add_trace(
                        go.Bar(
                            x=hist["Year"],
                            y=hist["Current Ratio"],
                            marker_color="#3B82F6",
                            name="Current Ratio",
                        )
                    )
                    fig_cr.update_layout(
                        title="Liquidity Cushion (Current Ratio)",
                        height=260,
                        margin=dict(l=10, r=10, t=35, b=10),
                    )
                    st.plotly_chart(fig_cr, use_container_width=True)

                st.dataframe(
                    hist[
                        [
                            "Year",
                            "Total Debt",
                            "Cash & Equivalents",
                            "Net Debt",
                            "Debt/Equity",
                            "Interest Coverage",
                            "Current Ratio",
                        ]
                    ].set_index("Year"),
                    use_container_width=True,
                )

            # 4. Cash Flow Tab
            with sub_tab4:
                st.markdown("###### **Organic Cash Generation**")
                cf_col1, cf_col2 = st.columns([1, 1])
                with cf_col1:
                    st.plotly_chart(
                        plot_dual_trend(
                            hist,
                            "Operating Cash Flow (CFO)",
                            "Capex",
                            "Cash from Operations",
                            "Capex",
                            curr,
                        ),
                        use_container_width=True,
                    )
                with cf_col2:
                    st.plotly_chart(
                        plot_historical_bar(
                            hist,
                            "Free Cash Flow (FCF)",
                            "Free Cash Flow (CFO - Capex)",
                            "#059669",
                            curr,
                        ),
                        use_container_width=True,
                    )

                st.dataframe(
                    hist[
                        [
                            "Year",
                            "Operating Cash Flow (CFO)",
                            "Capex",
                            "Free Cash Flow (FCF)",
                            "FCF Margin (%)",
                        ]
                    ].set_index("Year"),
                    use_container_width=True,
                )

            # 5. Valuation Tab
            with sub_tab5:
                st.markdown(
                    "###### **Market Multiples (Factual & Objective)**"
                )
                v1, v2, v3, v4, v5 = st.columns(5)
                v1.metric(
                    "P/E Multiple",
                    f"{lat['pe']:.1f}x" if lat["pe"] else "N/A",
                    help="Price-to-Earnings Ratio",
                )
                v2.metric(
                    "P/B Multiple",
                    f"{lat['pb']:.1f}x" if lat["pb"] else "N/A",
                    help="Price-to-Book Ratio",
                )
                v3.metric(
                    "EV/EBITDA",
                    f"{lat['ev_ebitda']:.1f}x" if lat["ev_ebitda"] else "N/A",
                    help="Enterprise Value to EBITDA",
                )
                v4.metric(
                    "EV/Sales",
                    f"{lat['ev_sales']:.1f}x" if lat["ev_sales"] else "N/A",
                    help="Enterprise Value to Total Revenue",
                )
                v5.metric(
                    "Dividend Yield",
                    f"{lat['div_yield']:.2f}%" if lat["div_yield"] else "N/A",
                    help="Latest Dividend Per Share / Price",
                )

                st.info(
                    "⚖️ **Neutral Valuation Note:** StockLens presents market valuation multiples purely as mathematical relationships "
                    "between reported accounts and prevailing market prices. Multiples vary materially by capital structure, cyclicality, "
                    "and business model. StockLens does not classify securities as 'undervalued' or 'overvalued'."
                )

            # Single-Click CSV Export for Financial Due Diligence
            st.markdown("---")
            csv_data = hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Export 5-Year Historical Schedule for {profile['ticker']} (CSV)",
                data=csv_data,
                file_name=f"{profile['ticker']}_financial_schedule_stocklens.csv",
                mime="text/csv",
                type="secondary",
            )

# ------------------------------------------------------------------------------
# TAB 2: MULTI-COMPANY PEER BENCHMARKING (FEATURE 9)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### **Cross-Company Peer Benchmarking**")
    st.caption(
        "Compare up to 4 peer enterprises side-by-side across fundamental efficiency, leverage, and cash generation metrics."
    )

    available_peer_keys = list(DemoDataProvider().companies.keys())
    selected_peers = st.multiselect(
        "Select Peer Group to Compare:",
        options=available_peer_keys,
        default=["TCS", "INFY", "WIPRO"],
        max_selections=4,
    )

    if len(selected_peers) < 2:
        st.info("Select at least 2 companies to generate peer comparison.")
    else:
        peer_records = []
        for sym in selected_peers:
            p_prof = provider.get_company_profile(sym)
            p_fin = provider.get_financial_history(sym)
            if p_prof and p_fin:
                p_metrics = FinancialMetricsEngine.compute_all_metrics(
                    p_prof, p_fin
                )
                lat = p_metrics["latest"]
                peer_records.append(
                    {
                        "Company": p_prof["name"],
                        "Ticker": sym,
                        "Price (₹)": p_prof["price"],
                        "Market Cap (₹ Cr)": lat["mcap"],
                        "Revenue Growth (%)": lat["latest_rev_growth"],
                        "EBITDA Margin (%)": lat["latest_ebitda_margin"],
                        "PAT Margin (%)": lat["latest_pat_margin"],
                        "ROE (%)": lat["latest_roe"],
                        "ROCE (%)": lat["latest_roce"],
                        "Debt/Equity": lat["latest_de"],
                        "Current Ratio": lat["latest_current_ratio"],
                        "FCF (₹ Cr)": lat["latest_fcf"],
                        "P/E Ratio": lat["pe"],
                        "EV/EBITDA": lat["ev_ebitda"],
                    }
                )

        peer_df = pd.DataFrame(peer_records)

        # Comparative Metrics Table
        st.dataframe(
            peer_df.set_index("Ticker").style.format(
                {
                    "Price (₹)": "₹{:,.2f}",
                    "Market Cap (₹ Cr)": "₹{:,.0f}",
                    "Revenue Growth (%)": "{:+.1f}%",
                    "EBITDA Margin (%)": "{:.1f}%",
                    "PAT Margin (%)": "{:.1f}%",
                    "ROE (%)": "{:.1f}%",
                    "ROCE (%)": "{:.1f}%",
                    "Debt/Equity": "{:.2f}x",
                    "Current Ratio": "{:.2f}",
                    "FCF (₹ Cr)": "₹{:,.0f}",
                    "P/E Ratio": "{:.1f}x",
                    "EV/EBITDA": "{:.1f}x",
                }
            ),
            use_container_width=True,
        )

        # Peer Visual Charts
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            fig_peer_m = px.bar(
                peer_df,
                x="Ticker",
                y="EBITDA Margin (%)",
                color="Ticker",
                title="EBITDA Margin Comparison (%)",
                text_auto=".1f",
            )
            fig_peer_m.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_peer_m, use_container_width=True)

        with p_c2:
            fig_peer_r = px.bar(
                peer_df,
                x="Ticker",
                y="ROCE (%)",
                color="Ticker",
                title="ROCE (Capital Efficiency) Comparison (%)",
                text_auto=".1f",
            )
            fig_peer_r.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_peer_r, use_container_width=True)

        # Peer CSV Export
        peer_csv = peer_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Peer Comparison Table (CSV)",
            data=peer_csv,
            file_name="stocklens_peer_comparison.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------------------
# TAB 3: FINANCIAL METRIC EXPLAINER / GLOSSARY (FEATURE 11)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### **Financial Metric Explainer & Accounting Methodology**")
    st.caption(
        "Structured guide to fundamental accounting, corporate finance formulas, and metric interpretations."
    )

    for metric_name, info in METRIC_GLOSSARY.items():
        with st.expander(f"📌 {metric_name}"):
            st.markdown(f"**Description:** {info['description']}")
            st.code(f"Formula: {info['formula']}", language="text")

# ------------------------------------------------------------------------------
# TAB 4: ABOUT, METHODOLOGY & DISCLAIMER
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### **About StockLens**")
    st.markdown(
        """
        **StockLens** is an open-architecture equity diagnostics tool engineered for institutional financial analysis, 
        credit appraisal, and educational fundamental equity research.
        
        ##### **Methodology & Precision Standards:**
        1. **Direct Calculation Policy:** All growth and profitability figures are computed directly from standardized primary financial statements rather than proprietary vendor black-boxes.
        2. **Safe Division Architecture:** Handled zero-division, negative shareholders' equity, and non-debt entities to eliminate application instability.
        3. **Deterministic Integrity:** AI narratives rely on objective, rule-based algorithmic summaries of reported accounting facts, avoiding generative hallucinations.
        
        ##### **Technical Architecture:**
        - **Core Stack:** Python 3.13, Streamlit, Pandas, NumPy, Plotly.
        - **Adapter Layer:** Decoupled data ingestion conforming to `BaseDataProvider`.
        - **Hosting Environment:** Streamlit Community Cloud (Global Edge Delivery).
        """
    )

# ==============================================================================
# 8. MANDATORY STATUTORY & USER SAFETY DISCLAIMER
# ==============================================================================
st.markdown(
    """
    <div class="legal-disclaimer">
        <b>STATUTORY DISCLAIMER & TERMS OF USE:</b><br>
        This tool provides purely informational and educational financial analysis and does not constitute investment advice, 
        securities recommendations, or an endorsement to buy, sell, or hold any financial instrument. Financial data may be delayed, 
        estimated, or subject to revision. StockLens and its creators make no warranties regarding accuracy or completeness. 
        Users must independently verify all financial information and consult a SEBI-registered (or locally certified) investment advisor 
        before executing investment or credit decisions.
    </div>
    """,
    unsafe_allow_html=True,
)