
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Global DCF Investor", page_icon="🌍", layout="wide")

st.title("🌍 Global DCF Investor")
st.caption("Global stock screener + normalized DCF valuation + scenario analysis")

# ---------------- Data layer ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker):
    t = yf.Ticker(ticker)
    return t.info, t.financials, t.balance_sheet, t.cashflow, t.history(period="5y", auto_adjust=False)

def val(df, labels, default=np.nan):
    if df is None or df.empty:
        return default
    for label in labels:
        if label in df.index:
            try:
                return float(df.loc[label].iloc[0])
            except Exception:
                pass
    return default

def latest_series(df, labels):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for label in labels:
        if label in df.index:
            s = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            if len(s):
                return s
    return pd.Series(dtype=float)

def money(x, currency=""):
    if pd.isna(x): return "N/A"
    ax = abs(x)
    if ax >= 1e12: s=f"{x/1e12:.2f}T"
    elif ax >= 1e9: s=f"{x/1e9:.2f}B"
    elif ax >= 1e6: s=f"{x/1e6:.1f}M"
    elif ax >= 1e3: s=f"{x/1e3:.1f}K"
    else: s=f"{x:.0f}"
    return f"{s} {currency}".strip()

def calc_dcf(fcf, growth, fade, wacc, terminal_growth, years, net_debt, shares):
    flows=[]
    x=fcf
    for i in range(years):
        x *= 1 + growth + fade*i
        flows.append(x)
    pv=[x/((1+wacc)**(i+1)) for i,x in enumerate(flows)]
    tv=flows[-1]*(1+terminal_growth)/(wacc-terminal_growth)
    pvtv=tv/((1+wacc)**years)
    ev=sum(pv)+pvtv
    eq=ev-net_debt
    per=eq/shares if shares else np.nan
    return per, ev, eq, flows, pv, tv

def score_company(info, financials, cashflow):
    fcf = val(cashflow, ["Operating Cash Flow","Total Cash From Operating Activities"]) + val(cashflow, ["Capital Expenditure","Capital Expenditures"], 0)
    rev = latest_series(financials, ["Total Revenue","Operating Revenue"])
    ni = val(financials, ["Net Income","Net Income Common Stockholders"])
    roe = info.get("returnOnEquity")
    margin = info.get("profitMargins")
    debt_to_equity = info.get("debtToEquity")
    growth = info.get("revenueGrowth")
    score=0
    parts={}
    parts["FCF positive"] = int(fcf > 0)
    parts["Revenue growth"] = int(growth is not None and growth > 0.03)
    parts["Profit margin"] = int(margin is not None and margin > 0.08)
    parts["ROE"] = int(roe is not None and roe > 0.10)
    parts["Debt discipline"] = int(debt_to_equity is not None and debt_to_equity < 150)
    score=sum(parts.values())
    return score, parts


# ---------------- Technical signals ----------------
def technical_signals(hist):
    if hist is None or hist.empty or len(hist) < 220:
        return pd.DataFrame(), {}
    df=hist.copy().dropna(subset=["Close"])
    close=df["Close"]; vol=df["Volume"].fillna(0)
    df["SMA50"]=close.rolling(50).mean(); df["SMA200"]=close.rolling(200).mean()
    delta=close.diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"]=100-(100/(1+(gain/loss.replace(0,np.nan))))
    df["AvgVol20"]=vol.rolling(20).mean(); df["RelVol20"]=vol/df["AvgVol20"]
    x=df.iloc[-1]
    trend="Bullish" if x["Close"]>x["SMA50"]>x["SMA200"] else ("Bearish" if x["Close"]<x["SMA50"]<x["SMA200"] else "Mixed")
    signals=[]
    if x["Close"]>x["SMA50"]: signals.append("Above 50-day MA")
    if x["Close"]>x["SMA200"]: signals.append("Above 200-day MA")
    if x["SMA50"]>x["SMA200"]: signals.append("50-day MA above 200-day MA")
    if 50<x["RSI14"]<70: signals.append("Positive RSI regime")
    if x["RelVol20"]>=1.5: signals.append("High relative volume")
    return df,{"trend":trend,"signals":signals,"rsi":x["RSI14"],"relvol":x["RelVol20"]}

def detect_cup_handle(hist):
    if hist is None or hist.empty or len(hist)<120: return None
    c=hist["Close"].dropna(); v=hist["Volume"].fillna(0)
    for n in [120,150,180,220]:
        if len(c)<n: continue
        s=c.iloc[-n:]; vs=v.loc[s.index]
        cup=s.iloc[:int(n*.80)]; handle=s.iloc[int(n*.72):]
        left_peak=s.iloc[:int(n*.55)].max()
        right_peak=cup.iloc[int(len(cup)*.65):].max()
        bottom=cup.min(); pos=int(np.argmin(cup.values))
        if not int(len(cup)*.25)<pos<int(len(cup)*.75): continue
        rim=(left_peak+right_peak)/2; depth=rim-bottom
        if rim<=0 or not .10<=depth/rim<=.45 or right_peak/rim<.90: continue
        handle_low=handle.min(); hd=(right_peak-handle_low)/depth
        if hd>.50: continue
        hv=vs.loc[handle.index].mean(); pv=vs.iloc[int(n*.50):int(n*.72)].mean()
        rel=float(v.iloc[-1]/max(v.rolling(20).mean().iloc[-1],1))
        breakout=float(s.iloc[-1])>float(right_peak)
        conf=(20 if right_peak/rim>=.97 else 10)+(20 if hd<=.33 else 10)+(15 if hv<pv else 0)+(20 if breakout else 0)+(15 if rel>=1.5 else (7 if rel>=1.2 else 0))+(10 if pos>int(len(cup)*.30) else 5)
        return {"confidence":min(100,conf),"rim":rim,"cup_low":bottom,"handle_low":handle_low,
                "depth_pct":depth/rim,"handle_depth_pct":hd,"breakout":breakout,
                "relative_volume":rel,"target":right_peak+depth,"stop":handle_low}
    return None


# ---------------- Expanded pattern engine ----------------
def atr(df, n=14):
    h,l,c=df["High"],df["Low"],df["Close"]
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pattern_engine(hist):
    if hist is None or hist.empty or len(hist)<80:
        return {}
    d=hist.dropna(subset=["Open","High","Low","Close"]).copy()
    d["SMA20"]=d["Close"].rolling(20).mean()
    d["SMA50"]=d["Close"].rolling(50).mean()
    d["SMA200"]=d["Close"].rolling(200).mean() if len(d)>=200 else np.nan
    d["ATR14"]=atr(d)
    d["Vol20"]=d["Volume"].rolling(20).mean()
    last=d.iloc[-1]
    out={}
    # Momentum / trend
    out["trend_bull"] = bool(last["Close"]>last["SMA50"] and (pd.isna(last["SMA200"]) or last["SMA50"]>last["SMA200"]))
    out["rsi"] = float((100-(100/(1+((d["Close"].diff().clip(lower=0).rolling(14).mean())/
                                     (-d["Close"].diff().clip(upper=0).rolling(14).mean()).replace(0,np.nan))))).iloc[-1])
    out["relvol"] = float(last["Volume"]/last["Vol20"]) if last["Vol20"] else 0
    # Simple double-bottom candidate: two similar lows separated by a rally.
    w=d.tail(min(160,len(d)))
    lows=w["Low"].rolling(7,center=True).min()
    idx=w.index[(w["Low"]==lows)&lows.notna()]
    out["double_bottom"]=False
    if len(idx)>=2:
        a,b=idx[-2],idx[-1]
        pa,pb=w.loc[a,"Low"],w.loc[b,"Low"]
        if abs(pa-pb)/max(pa,pb)<.06 and (b-a).days>=10:
            between=w.loc[a:b,"High"].max()
            if between/min(pa,pb)>1.08 and last["Close"]>min(pa,pb):
                out["double_bottom"]=True
    # Ascending triangle: rising lows + relatively flat resistance in last 60 bars.
    z=d.tail(60)
    resistance=z["High"].quantile(.9)
    slope=np.polyfit(np.arange(len(z)),z["Low"].values,1)[0]
    out["ascending_triangle"]=bool(slope>0 and (z["High"].max()-z["High"].quantile(.8))/z["High"].max()<.08 and last["Close"]<resistance*1.04)
    # Head-and-shoulders candidate: three local peaks, middle highest, shoulders similar.
    out["head_shoulders"]=False
    peaks=z["High"].rolling(7,center=True).max()
    pi=list(z.index[(z["High"]==peaks)&peaks.notna()])
    if len(pi)>=3:
        p1,p2,p3=pi[-3:]
        h1,h2,h3=z.loc[p1,"High"],z.loc[p2,"High"],z.loc[p3,"High"]
        if h2>h1*1.04 and h2>h3*1.04 and abs(h1-h3)/max(h1,h3)<.08:
            out["head_shoulders"]=True
    # Flags: tight consolidation after a strong recent move.
    if len(d)>=35:
        pre=d.iloc[-35:-15]["Close"]; recent=d.iloc[-15:]["Close"]
        move=pre.iloc[-1]/pre.iloc[0]-1
        cons=(recent.max()-recent.min())/recent.mean()
        out["bull_flag"]=bool(move>.12 and cons<.10 and last["Close"]>=recent.mean())
        out["bear_flag"]=bool(move<-.12 and cons<.10 and last["Close"]<=recent.mean())
    else:
        out["bull_flag"]=out["bear_flag"]=False
    # Composite technical score, not a prediction.
    score=50
    score += 15 if out["trend_bull"] else -15
    score += 10 if 50<=out["rsi"]<=70 else (-5 if out["rsi"]>70 else 0)
    score += 10 if out["relvol"]>=1.5 else 0
    score += 8 if out["double_bottom"] else 0
    score += 8 if out["ascending_triangle"] else 0
    score += 8 if out["bull_flag"] else 0
    score -= 12 if out["head_shoulders"] else 0
    score -= 8 if out["bear_flag"] else 0
    out["score"]=int(max(0,min(100,score)))
    return out

def historical_signal_test(hist, horizon=20):
    # Lightweight walk-forward test of a simple breakout signal:
    # close above 20-day high + volume > 1.5x 20-day average.
    if hist is None or len(hist)<100: return None
    d=hist.copy()
    d["High20"]=d["High"].rolling(20).max().shift(1)
    d["Vol20"]=d["Volume"].rolling(20).mean()
    d["Signal"]=(d["Close"]>d["High20"])&(d["Volume"]>1.5*d["Vol20"])
    future=d["Close"].shift(-horizon)/d["Close"]-1
    sig=future[d["Signal"]].dropna()
    if len(sig)==0: return {"trades":0}
    return {"trades":int(len(sig)),"win_rate":float((sig>0).mean()),"avg_return":float(sig.mean()),
            "median_return":float(sig.median())}


def investment_rating(per_share, price, fundamental_score, technical_score):
    """Combine valuation, fundamentals and technicals into a simple 3-level screen."""
    if pd.isna(per_share) or pd.isna(price) or price <= 0:
        return "N/A", np.nan, "DCF value unavailable"
    upside = per_share / price - 1

    # DCF component: rewards meaningful upside, penalizes overvaluation.
    if upside <= -0.20:
        dcf_score = 0
    elif upside < 0:
        dcf_score = 25 + (upside + 0.20) / 0.20 * 25
    elif upside < 0.10:
        dcf_score = 50 + upside / 0.10 * 15
    elif upside < 0.25:
        dcf_score = 65 + (upside - 0.10) / 0.15 * 15
    elif upside < 0.50:
        dcf_score = 80 + (upside - 0.25) / 0.25 * 15
    else:
        dcf_score = 95

    fundamental_score_100 = (fundamental_score / 5) * 100
    technical_score_100 = float(technical_score if not pd.isna(technical_score) else 50)

    # Balanced weighting: DCF 40%, fundamentals 30%, technicals 30%.
    composite = 0.40 * dcf_score + 0.30 * fundamental_score_100 + 0.30 * technical_score_100

    if composite >= 70:
        rating = "BUY"
    elif composite >= 50:
        rating = "HOLD"
    else:
        rating = "AVOID"

    return rating, composite, f"DCF upside {upside*100:.1f}%"

# ---------------- Sidebar ----------------
st.sidebar.header("Universe")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
st.sidebar.caption("International examples: VOD.L, SAP.DE, ASML.AS, TSM, 7203.T, BHP.AX, RELIANCE.NS")

st.sidebar.header("DCF")
fcf_mode = st.sidebar.radio("Base FCF", ["Automatic latest FCF", "Manual normalized FCF"])
growth = st.sidebar.slider("Starting FCF growth", -20.0, 40.0, 10.0, .5)/100
fade = st.sidebar.slider("Annual growth fade", -5.0, 5.0, -1.0, .25)/100
years = st.sidebar.slider("Forecast years", 3, 15, 10)
wacc = st.sidebar.slider("WACC", 4.0, 20.0, 9.0, .25)/100
tg = st.sidebar.slider("Terminal growth", 0.0, 5.0, 2.5, .1)/100
mos = st.sidebar.slider("Margin of safety", 0.0, 50.0, 20.0, 1.0)/100

if wacc <= tg:
    st.error("WACC must be greater than terminal growth.")
    st.stop()

try:
    info, fin, bal, cf, hist = load_data(ticker)
except Exception as e:
    st.error(f"Unable to load {ticker}: {e}")
    st.stop()

name = info.get("longName") or info.get("shortName") or ticker
currency = info.get("currency","USD")
price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan
shares = info.get("sharesOutstanding") or np.nan
ocf = val(cf, ["Operating Cash Flow","Total Cash From Operating Activities"])
capex = val(cf, ["Capital Expenditure","Capital Expenditures"], 0)
auto_fcf = ocf + capex if not pd.isna(ocf) else np.nan
cash = val(bal, ["Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments"], 0)
debt = val(bal, ["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"], 0)
net_debt = debt - cash
if fcf_mode == "Manual normalized FCF":
    base_fcf = st.sidebar.number_input("Normalized FCF", min_value=0.0, value=float(auto_fcf if not pd.isna(auto_fcf) and auto_fcf>0 else 1e9), step=1e8)
else:
    base_fcf = auto_fcf

# ---------------- Dashboard ----------------
st.subheader(f"{name} ({ticker})")
a,b,c,d,e = st.columns(5)
a.metric("Price", f"{price:,.2f} {currency}" if not pd.isna(price) else "N/A")
b.metric("FCF", money(base_fcf,currency))
c.metric("Revenue", money(val(fin,["Total Revenue","Operating Revenue"]),currency))
d.metric("Net debt", money(net_debt,currency))
e.metric("Country", info.get("country","N/A"))

if pd.isna(base_fcf) or base_fcf <= 0:
    st.warning("FCF is missing or negative. Switch to Manual normalized FCF and enter owner earnings / normalized FCF.")

if not pd.isna(base_fcf) and base_fcf > 0 and not pd.isna(shares) and shares > 0:
    per, ev, eq, flows, pv, tv = calc_dcf(base_fcf,growth,fade,wacc,tg,years,net_debt,shares)
    mos_value = per*(1-mos)
else:
    per=ev=eq=mos_value=np.nan
    flows=[]; pv=[]; tv=np.nan

# ---------------- Combined investment rating ----------------
fundamental_score, fundamental_parts = score_company(info, fin, cf)
pattern_snapshot = pattern_engine(hist)
technical_score = pattern_snapshot.get("score", 50) if pattern_snapshot else 50
rating, composite_score, rating_note = investment_rating(per, price, fundamental_score, technical_score)

st.divider()
st.subheader("Investment rating")
r1, r2, r3, r4 = st.columns(4)
if rating == "BUY":
    r1.success("🟢 BUY")
elif rating == "HOLD":
    r1.warning("🟡 HOLD")
elif rating == "AVOID":
    r1.error("🔴 AVOID")
else:
    r1.metric("Rating", "N/A")
r2.metric("Overall score", f"{composite_score:.0f}/100" if not pd.isna(composite_score) else "N/A")
r3.metric("Fundamentals", f"{fundamental_score}/5")
r4.metric("Technical", f"{technical_score}/100")
st.caption("Balanced screen: DCF valuation 40% + fundamentals 30% + technicals 30%. This is a screening aid, not personal investment advice.")

st.divider()
st.subheader("Intrinsic value")
m1,m2,m3,m4 = st.columns(4)
m1.metric("DCF value / share", f"{per:,.2f} {currency}" if not pd.isna(per) else "N/A")
m2.metric("MOS value / share", f"{mos_value:,.2f} {currency}" if not pd.isna(mos_value) else "N/A")
upside=(per/price-1) if price and not pd.isna(price) and not pd.isna(per) else np.nan
mos_up=(mos_value/price-1) if price and not pd.isna(price) and not pd.isna(mos_value) else np.nan
m3.metric("DCF upside", f"{upside*100:.1f}%" if not pd.isna(upside) else "N/A")
m4.metric("MOS upside", f"{mos_up*100:.1f}%" if not pd.isna(mos_up) else "N/A")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Forecast","Sensitivity","Quality","Price history","Trading signals"])

with tab1:
    if flows:
        forecast=pd.DataFrame({"Year":range(1,years+1),"FCF":flows,"PV FCF":pv})
        st.dataframe(forecast, use_container_width=True)
        st.bar_chart(forecast.set_index("Year")[["FCF"]])

with tab2:
    gs=np.arange(max(-.10,growth-.05), min(.40,growth+.051), .01)
    ws=np.arange(max(.04,wacc-.04), min(.20,wacc+.041), .01)
    mat=[]
    for g in gs:
        row=[]
        for w in ws:
            if w <= tg: row.append(np.nan); continue
            p,*_=calc_dcf(base_fcf,g,fade,w,tg,years,net_debt,shares)
            row.append(p)
        mat.append(row)
    sens=pd.DataFrame(mat,index=[f"{x*100:.0f}%" for x in gs],columns=[f"{x*100:.0f}%" for x in ws])
    sens.index.name="Growth ↓ / WACC →"
    st.dataframe(sens, use_container_width=True)

with tab3:
    score,parts=score_company(info,fin,cf)
    st.metric("Fundamental quality score",f"{score}/5")
    q=pd.DataFrame({"Test":list(parts.keys()),"Pass":[("✓" if x else "—") for x in parts.values()]})
    st.table(q)
    st.write({
        "ROE": info.get("returnOnEquity"),
        "Profit margin": info.get("profitMargins"),
        "Revenue growth": info.get("revenueGrowth"),
        "Debt/equity": info.get("debtToEquity"),
        "Beta": info.get("beta"),
    })

with tab4:
    if hist is not None and not hist.empty:
        st.line_chart(hist["Close"])
    else:
        st.info("No price history available.")

with tab5:
    tech, summary = technical_signals(hist)
    patterns=pattern_engine(hist)
    bt=historical_signal_test(hist)

    cup=detect_cup_handle(hist)
    st.subheader("Technical trading signals")
    a,b,c,d=st.columns(4)
    a.metric("Trend",summary.get("trend","N/A"))
    b.metric("RSI(14)",f"{summary.get("rsi",np.nan):.1f}")
    c.metric("Relative volume",f"{summary.get("relvol",np.nan):.2f}x")
    d.metric("Technical signals",len(summary.get("signals",[])))
    if patterns:
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Composite technical score",f"{patterns['score']}/100")
        p2.metric("Double bottom","Candidate" if patterns["double_bottom"] else "—")
        p3.metric("Ascending triangle","Candidate" if patterns["ascending_triangle"] else "—")
        p4.metric("Head & shoulders","Candidate" if patterns["head_shoulders"] else "—")
        p5,p6=st.columns(2)
        p5.metric("Bull flag","Candidate" if patterns["bull_flag"] else "—")
        p6.metric("Bear flag","Candidate" if patterns["bear_flag"] else "—")
    if bt and bt.get("trades",0):
        st.caption(f"Historical diagnostic: {bt['trades']} breakout events in the available history; "
                   f"{bt['win_rate']*100:.1f}% positive after 20 sessions; average return {bt['avg_return']*100:.1f}%. "
                   "This is descriptive, not a forecast.")
    if cup:
        st.success(f"Potential cup-and-handle candidate — confidence {cup['confidence']}/100")
        a,b,c,d=st.columns(4)
        a.metric("Breakout","Confirmed" if cup["breakout"] else "Not confirmed")
        b.metric("Rim / breakout",f"{cup['rim']:,.2f}")
        c.metric("Measured target",f"{cup['target']:,.2f}")
        d.metric("Handle stop reference",f"{cup['stop']:,.2f}")
        st.write(f"Cup depth {cup['depth_pct']*100:.1f}% · Handle depth {cup['handle_depth_pct']*100:.1f}% · Breakout relative volume {cup['relative_volume']:.2f}x")
    else:
        st.info("No sufficiently clean cup-and-handle candidate detected in the available history.")
    st.write("**Other signals:** "+(", ".join(summary.get("signals",[])) if summary.get("signals") else "None"))
    if not tech.empty: st.line_chart(tech[["Close","SMA50","SMA200"]].tail(252))

# ---------------- Watchlist / screener ----------------
st.divider()
st.subheader("Build a watchlist")
st.write("Enter comma-separated tickers. The app will calculate a comparable DCF and quality score for each.")
watch = st.text_input("Watchlist", "AAPL,MSFT,GOOGL,AMZN,NVDA,ASML.AS,SAP.DE,VOD.L,7203.T,BHP.AX")
run = st.button("Value watchlist", type="primary")

if run:
    rows=[]
    for tkr in [x.strip().upper() for x in watch.split(",") if x.strip()]:
        try:
            i,f,b,cf2,h=load_data(tkr)
            cup2=detect_cup_handle(h)
            pat2=pattern_engine(h)
            p=i.get("currentPrice") or i.get("regularMarketPrice") or np.nan
            sh=i.get("sharesOutstanding") or np.nan
            od=val(b,["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"],0)
            ca=val(b,["Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments"],0)
            fcf=val(cf2,["Operating Cash Flow","Total Cash From Operating Activities"]) + val(cf2,["Capital Expenditure","Capital Expenditures"],0)
            if pd.isna(fcf) or fcf<=0 or pd.isna(sh) or sh<=0:
                rows.append({"Ticker":tkr,"Company":i.get("shortName",tkr),"Price":p,"DCF":np.nan,"MOS":np.nan,"Upside %":np.nan,"Quality":score_company(i,f,cf2)[0],"Cup&Handle":"Candidate" if cup2 else "—","Tech confidence":cup2["confidence"] if cup2 else 0,"Tech score":pat2.get("score",50) if pat2 else 50})
                continue
            pp,*_=calc_dcf(fcf,growth,fade,wacc,tg,years,od-ca,sh)
            rows.append({"Ticker":tkr,"Company":i.get("shortName",tkr),"Price":p,"DCF":pp,"MOS":pp*(1-mos),"Upside %":(pp/p-1)*100 if p else np.nan,"Quality":score_company(i,f,cf2)[0],"Cup&Handle":"Candidate" if cup2 else "—","Tech confidence":cup2["confidence"] if cup2 else 0,"Tech score":pat2.get("score",50) if pat2 else 50})
        except Exception as ex:
            rows.append({"Ticker":tkr,"Company":"Error","Price":np.nan,"DCF":np.nan,"MOS":np.nan,"Upside %":np.nan,"Quality":0,"Cup&Handle":"—","Tech confidence":0})
    out=pd.DataFrame(rows)
    out["Opportunity score"] = (
        out["Upside %"].clip(-50,100).fillna(-50).add(50).div(150).mul(50)
        + out["Quality"].fillna(0).div(5).mul(25)
        + out.get("Tech score",pd.Series(50,index=out.index)).fillna(50).div(100).mul(25)
    )
    out=out.sort_values(["Opportunity score","Upside %"],ascending=False)
    st.dataframe(out, use_container_width=True)

st.caption("Educational valuation model. Verify financial statements, corporate actions, share counts, FX, debt/cash and assumptions before making investment decisions.")
