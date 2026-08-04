# FIELDS_USED.md — final Bloomberg mnemonics for Track B (spec 4B.1)

Session: 2026-08-04, one sitting, NVDA/AAPL used for FLDS verification.
Rule applied: FLDS beats any pre-drafted guess; substitutions recorded below.
This file is cited by leakage-audit item 10 and the limitations section.

Pulls taken: **quarterly** (Per=CQ — the committed Track B input, per config),
plus **monthly** (Per=CM) and **daily** (Per=CD) workbooks for the
validation-only refresh-cadence robustness check. The grid uses quarterly.

## Final field list — all 19 confirmed 2026-08-04

| # | Concept | Mnemonic used | Note |
|---|---|---|---|
| 1 | P/E | `PE_RATIO` | as drafted |
| 2 | Price/Book | `PX_TO_BOOK_RATIO` | as drafted |
| 3 | Price/Sales | `PX_TO_SALES_RATIO` | as drafted |
| 4 | Price/EBITDA | `PX_TO_EBITDA` | as drafted (verified in FLDS) |
| 5 | Market cap | `CUR_MKT_CAP` | as drafted |
| 6 | Shares outstanding | `EQY_SH_OUT` | as drafted |
| 7 | Sales growth | `SALES_GROWTH` | as drafted |
| 8 | Cash-flow growth | `CASH_FLOW_GROWTH` | as drafted (verified in FLDS) |
| 9 | **FCF growth** | **`FREE_CASH_FLOW_1_YEAR_GROWTH`** | **SUBSTITUTED.** Drafted `FREE_CASH_FLOW_GROWTH` does not exist in FLDS. Chose "Free Cash Flow 1 Year Growth": whole-firm FCF, year-over-year — matches the 1-yr horizon of the other growth fields. Rejected: per-share variants (buyback-contaminated), 3/5-yr CAGRs (wrong horizon), sequential QoQ (noisy/seasonal). Designated backup was `Free Cash Flow to Firm 1 Year Growth`. |
| 10 | Normalized ROE | `RETURN_COM_EQY` | as drafted (verified in FLDS) |
| 11 | **Dividend/share** | **`EQY_DVD_SH_12M_NET`** | **SUBSTITUTED.** Drafted `EQY_DVD_SH` does not exist in FLDS. Chose "Dividend Per Share 12 Month (Net)": trailing-12M level smooths payout timing; net≈gross for US large-caps. Rejected: `EQY_DVD_YLD_12M` (a yield, different concept — price already carried by valuation ratios), `IS_DIV_PER_SHR` (per-fiscal-period, lumpy), `EQY_DPS` (ambiguous window). Zero/near-zero values for low-payers (e.g. NVDA early years) are real data, not gaps. |
| 12 | Volatility 60d | `VOLATILITY_60D` | as drafted |
| 13 | RSI | `RSI_14D` | as drafted |
| 14 | Close | `PX_LAST` | as drafted |
| 15 | Ask | `PX_ASK` | as drafted |
| 16 | Bid | `PX_BID` | as drafted |
| 17 | Analyst rating | `TOT_ANALYST_REC` | as drafted (verified in FLDS) |
| 18 | Buy recommendations | `TOT_BUY_REC` | as drafted |
| 19 | Sell recommendations | `TOT_SELL_REC` | as drafted |

CSV naming: one file per field, named exactly after the mnemonic
(e.g. `FREE_CASH_FLOW_1_YEAR_GROWTH.csv`, `EQY_DVD_SH_12M_NET.csv`);
monthly/daily variants suffixed `_monthly` / `_daily` where exported as CSV
(otherwise they live as cached values in `characteristics_monthly.xlsx` /
`characteristics_daily.xlsx`).

Pull parameters: `12/31/2014 – 12/31/2024`, `Days=A`, `Fill=P`
(daily workbook: `Days=T` where accepted).

## Point-in-time note (audit item 10)

Fields were pulled with Bloomberg's standard (potentially restated)
definitions; no dedicated as-reported variants were substituted during the
session. Per spec 4B.1 this goes on the limitations disclosure list — the
column stubs in `docs/limitations.md` already carry it.

## Dropped fields (coverage <~90% or unusable history)

None — all 19 pulled.
