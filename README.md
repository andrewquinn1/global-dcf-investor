# Global DCF Investor — Pro MVP

## Included

- Global ticker input using Yahoo Finance ticker suffixes.
- DCF based on free cash flow.
- Manual normalized FCF option.
- Growth fade, WACC, terminal growth and margin of safety.
- DCF intrinsic value and upside/downside.
- Forecast table and chart.
- Growth/WACC sensitivity matrix.
- Basic fundamental quality score.
- 5-year price chart.
- Multi-company watchlist/screener with comparable DCF values and quality scores.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important production upgrade

For a serious commercial global product, replace the Yahoo Finance data layer with a licensed fundamentals provider and retain source/filing provenance. Yahoo's own help page states that its data is for informational purposes and should not be redistributed; therefore this repository is best treated as a prototype rather than a commercial data service.

For global coverage, a paid fundamentals provider should be selected after testing IFRS/GAAP normalization, historical depth, restatements, corporate actions and licensing. Current 2026 market comparisons show meaningful differences between providers in global coverage and source traceability.


## Technical signals

Added heuristic 50/200-day trend, RSI, relative-volume and cup-and-handle candidate detection, including breakout confirmation, measured target and handle-low reference. The detector is deliberately a screening aid rather than a guarantee of a valid pattern; classic definitions emphasize a prior uptrend, rounded cup, shallower handle and volume-backed breakout. citeturn0search0turn0search10


## Expanded technical engine

Added:
- Composite technical score
- Double-bottom candidate
- Ascending-triangle candidate
- Bull/bear flag candidates
- Head-and-shoulders candidate
- 20-day breakout + 1.5x-volume historical diagnostic
- Watchlist technical score
- Combined Opportunity Score blending valuation, fundamental quality and technical setup

The pattern engine is intentionally conservative and heuristic. Pattern recognition is probabilistic and can produce false positives. The cup-and-handle logic follows common definitions emphasizing a rounded cup, shallower handle, resistance breakout and stronger volume. citeturn0search0turn0search1
