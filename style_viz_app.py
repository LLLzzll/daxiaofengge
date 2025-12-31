import os
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

def render_echarts(option, height=420):
    try:
        from streamlit_echarts import st_echarts
        st_echarts(options=option, height=height)
    except Exception:
        html = f"""
        <div id="echarts_container" style="width:100%;height:{height}px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        <script>
        var chart = echarts.init(document.getElementById('echarts_container'));
        var option = {json.dumps(option)};
        chart.setOption(option);
        </script>
        """
        st.components.v1.html(html, height=height + 50)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def to_date_str(x):
    try:
        return str(x)[:10]
    except Exception:
        return str(x)

def load_boundaries():
    p = os.path.join("data", "jq_output", "boundaries.csv")
    df = safe_read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["date", "large_lower", "small_upper"])
    cols = df.columns.tolist()
    date_col = None
    for c in ["date", "dealDate", "time", "datetime"]:
        if c in cols:
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]
    ll = None
    su = None
    for c in ["large_lower", "large_threshold", "large_bound", "large"]:
        if c in cols:
            ll = c
            break
    for c in ["small_upper", "small_threshold", "small_bound", "small"]:
        if c in cols:
            su = c
            break
    if ll is None:
        ll = cols[1] if len(cols) > 1 else cols[0]
    if su is None:
        su = cols[2] if len(cols) > 2 else ll
    df = df[[date_col, ll, su]].copy()
    df.columns = ["date", "large_lower", "small_upper"]
    df["date"] = df["date"].apply(to_date_str)
    df = df.sort_values("date")
    return df

def load_daily_group_returns():
    rows = []
    for grp in ["small", "large"]:
        base = os.path.join("data", "jq_output", grp)
        files = sorted(glob.glob(os.path.join(base, "*.csv")))
        for f in files:
            d = os.path.splitext(os.path.basename(f))[0]
            df = safe_read_csv(f)
            if df.empty:
                continue
            ret_col = None
            for c in ["pct_change", "change_pct", "ret"]:
                if c in df.columns:
                    ret_col = c
                    break
            if ret_col is None:
                continue
            m = pd.to_numeric(df[ret_col], errors="coerce").mean()
            rows.append({"date": d, "group": grp, "ret": m})
    if not rows:
        return pd.DataFrame(columns=["date", "large_ret", "small_ret", "spread"])
    x = pd.DataFrame(rows)
    piv = x.pivot(index="date", columns="group", values="ret")
    piv = piv.rename(columns={"large": "large_ret", "small": "small_ret"})
    piv["spread"] = piv.get("large_ret", np.nan) - piv.get("small_ret", np.nan)
    piv = piv.reset_index().rename(columns={"index": "date"}).sort_values("date")
    return piv

def line_option(x, series_list, title_text, yfmt="{value}", ymin=None, ymax=None, yint=None, y_split_line_show=True):
    yaxis = {"type": "value", "axisLabel": {"formatter": yfmt}, "splitLine": {"show": y_split_line_show}}
    if ymin is not None:
        yaxis["min"] = ymin
    if ymax is not None:
        yaxis["max"] = ymax
    if yint is not None:
        yaxis["interval"] = yint
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 40},
        "xAxis": {"type": "category", "data": x},
        "yAxis": yaxis,
        "grid": {"top": 80, "left": 60, "right": 40, "bottom": 50},
        "series": series_list
    }

def log_line_option(x, series_list, title_text, ymin=0.5, ymax=4.0, yfmt="{value}%", log_base=2):
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 40},
        "xAxis": {"type": "category", "data": x},
        "yAxis": {"type": "log", "logBase": log_base, "min": ymin, "max": ymax, "axisLabel": {"formatter": yfmt}},
        "grid": {"top": 80, "left": 60, "right": 40, "bottom": 50},
        "series": series_list
    }

def fixed_marklines(values, fmt="{value}%"):
    return {
        "silent": True,
        "symbol": "none",
        "label": {"formatter": fmt},
        "lineStyle": {"type": "dashed", "width": 1, "color": "#999"},
        "data": [{"yAxis": float(v)} for v in values]
    }

def bar_option(x, y, title_text):
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x},
        "yAxis": {"type": "value"},
        "grid": {"top": 60, "left": 60, "right": 40, "bottom": 50},
        "series": [{"name": "值", "type": "bar", "data": y}]
    }

def hist_option(x, y, title_text, vlines=None):
    ml = []
    if vlines:
        for v in vlines:
            if isinstance(v, str):
                lab = v if v in x else None
            else:
                a = str(v)
                b = f"{float(v):.2f}"
                lab = a if a in x else (b if b in x else None)
            if lab is not None:
                ml.append({"xAxis": lab})
    data_items = []
    for val in y:
        show_lbl = (val is not None) and (pd.notna(val)) and (float(val) != 0.0)
        data_items.append({"value": val, "label": {"show": show_lbl, "position": "top"}})
    series = {
        "name": "频数",
        "type": "bar",
        "data": data_items
    }
    if ml:
        series["markLine"] = {
            "silent": True,
            "symbol": "none",
            "label": {"formatter": "{c}"},
            "lineStyle": {"type": "solid", "width": 1, "color": "#aaa"},
            "data": ml
        }
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x, "axisLabel": {"interval": 0, "rotate": 35, "fontSize": 10, "showMinLabel": True, "showMaxLabel": True}},
        "yAxis": {"type": "value"},
        "grid": {"top": 60, "left": 60, "right": 40, "bottom": 80},
        "series": [series]
    }

def scatter_option(x, y, title_text, xlabel, ylabel):
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {},
        "xAxis": {"type": "value", "name": xlabel},
        "yAxis": {"type": "value", "name": ylabel},
        "grid": {"top": 60, "left": 60, "right": 40, "bottom": 50},
        "series": [{"type": "scatter", "data": [[float(a), float(b)] for a, b in zip(x, y)]}]
    }

def quadrant_ratios(x_vals, y_vals):
    sx = pd.Series(pd.to_numeric(x_vals, errors="coerce"))
    sy = pd.Series(pd.to_numeric(y_vals, errors="coerce"))
    mask = sx.notna() & sy.notna() & (sx != 0) & (sy != 0)
    if mask.sum() == 0:
        return {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "total": 0}
    sx = sx[mask]
    sy = sy[mask]
    total = int(len(sx))
    q1 = int(((sx > 0) & (sy > 0)).sum())
    q2 = int(((sx < 0) & (sy > 0)).sum())
    q3 = int(((sx < 0) & (sy < 0)).sum())
    q4 = int(((sx > 0) & (sy < 0)).sum())
    def pct(n):
        return round(n * 100.0 / total, 2)
    return {"q1": pct(q1), "q2": pct(q2), "q3": pct(q3), "q4": pct(q4), "total": total}

def load_intraday_group(date_str, grp):
    p = os.path.join("data", "jq_output_min15", grp, f"{date_str}.csv")
    df = safe_read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["time", "ret"])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(["code", "time"])
    if "close" not in df.columns:
        return pd.DataFrame(columns=["time", "ret"])
    if "open" in df.columns:
        oc = pd.to_numeric(df["open"], errors="coerce")
        cc = pd.to_numeric(df["close"], errors="coerce")
        df["ret"] = np.where(oc != 0, (cc / oc) - 1.0, np.nan)
    else:
        df["ret"] = df.groupby("code")["close"].pct_change()
    g = df.groupby("time")["ret"].mean().reset_index()
    g["time"] = g["time"].dt.strftime("%H:%M")
    return g

def intraday_option(date_str, small_df, large_df):
    x = sorted(list(set(small_df["time"].tolist() + large_df["time"].tolist())))
    s_map = {t: np.nan for t in x}
    l_map = {t: np.nan for t in x}
    for _, r in small_df.iterrows():
        s_map[r["time"]] = r["ret"]
    for _, r in large_df.iterrows():
        l_map[r["time"]] = r["ret"]
    s = [round(float(s_map[t]) * 100, 2) if pd.notna(s_map[t]) else None for t in x]
    l = [round(float(l_map[t]) * 100, 2) if pd.notna(l_map[t]) else None for t in x]
    sp = [round((float(l_map[t]) - float(s_map[t])) * 100, 2) if pd.notna(s_map[t]) and pd.notna(l_map[t]) else None for t in x]
    series = [
        {"name": "小盘15分钟平均收益率(%)", "type": "line", "smooth": True, "data": s, "connectNulls": True},
        {"name": "大盘15分钟平均收益率(%)", "type": "line", "smooth": True, "data": l, "connectNulls": True},
        {"name": "分钟Spread(%)", "type": "line", "smooth": True, "data": sp, "connectNulls": True, "areaStyle": {"opacity": 0.12}, "lineStyle": {"width": 2}},
    ]
    return line_option(x, series, f"{date_str} 15分钟平均收益率", "{value}%")

def minute_profile_last_month():
    base_small = os.path.join("data", "jq_output_min15", "small")
    base_large = os.path.join("data", "jq_output_min15", "large")
    files = sorted(glob.glob(os.path.join(base_small, "*.csv")))
    if not files:
        return pd.DataFrame(columns=["time", "spread"])
    rows = []
    for f in files:
        d = os.path.splitext(os.path.basename(f))[0]
        s = load_intraday_group(d, "small")
        l = load_intraday_group(d, "large")
        if s.empty or l.empty:
            continue
        m = pd.merge(s, l, on="time", suffixes=("_small", "_large"))
        m["spread"] = m["ret_large"] - m["ret_small"]
        rows.append(m[["time", "spread"]])
    if not rows:
        return pd.DataFrame(columns=["time", "spread"])
    x = pd.concat(rows)
    prof = x.groupby("time")["spread"].mean().reset_index()
    return prof

def index_daily_returns(path):
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "ret"])
    if "time" in df.columns:
        idx_col = "time"
    elif "datetime" in df.columns:
        idx_col = "datetime"
    else:
        idx_col = df.columns[0]
    if idx_col not in df.columns or "close" not in df.columns or "open" not in df.columns:
        return pd.DataFrame(columns=["date", "ret"])
    df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce")
    df["date"] = df[idx_col].dt.strftime("%Y-%m-%d")
    oc = pd.to_numeric(df["open"], errors="coerce")
    cc = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = np.where(oc != 0, (cc / oc - 1.0) * 100.0, np.nan)
    g = df.groupby("date")["ret"].mean().reset_index()
    return g[["date", "ret"]]

def list_all_codes():
    codes = set()
    for grp in ["small", "large"]:
        base = os.path.join("data", "jq_output", grp)
        files = sorted(glob.glob(os.path.join(base, "*.csv")))
        for f in files:
            df = safe_read_csv(f)
            if df.empty or "code" not in df.columns:
                continue
            vals = df["code"].dropna().astype(str).tolist()
            for c in vals:
                codes.add(c)
    return sorted(list(codes))

def stock_daily_returns(code):
    rows = []
    if not code:
        return pd.DataFrame(columns=["date", "ret"])
    for grp in ["small", "large"]:
        base = os.path.join("data", "jq_output", grp)
        files = sorted(glob.glob(os.path.join(base, "*.csv")))
        for f in files:
            d = os.path.splitext(os.path.basename(f))[0]
            df = safe_read_csv(f)
            if df.empty or "code" not in df.columns:
                continue
            ret_col = None
            for c in ["pct_change", "change_pct", "ret"]:
                if c in df.columns:
                    ret_col = c
                    break
            if ret_col is None:
                continue
            s = df[df["code"].astype(str) == str(code)]
            if s.empty:
                continue
            v = pd.to_numeric(s[ret_col], errors="coerce")
            if len(v) == 0:
                continue
            rows.append({"date": d, "ret": float(v.iloc[0])})
    if not rows:
        return pd.DataFrame(columns=["date", "ret"])
    x = pd.DataFrame(rows).dropna().sort_values("date")
    return x

def style_judgment(a_df, b_df, threshold=5):
    if a_df is None or b_df is None:
        return pd.DataFrame(columns=["date", "streak_a", "streak_b", "flag"])
    a = a_df[["date", "ret"]].rename(columns={"ret": "ret_a"})
    b = b_df[["date", "ret"]].rename(columns={"ret": "ret_b"})
    m = pd.merge(a, b, on="date", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["date", "streak_a", "streak_b", "flag"])
    m["ret_a"] = pd.to_numeric(m["ret_a"], errors="coerce")
    m["ret_b"] = pd.to_numeric(m["ret_b"], errors="coerce")
    m = m.dropna().sort_values("date")
    wins_a = (m["ret_a"] > m["ret_b"]).astype(int).tolist()
    wins_b = (m["ret_b"] > m["ret_a"]).astype(int).tolist()
    sa = []
    sb = []
    ca = 0
    cb = 0
    for i in range(len(wins_a)):
        ca = ca + 1 if wins_a[i] == 1 else 0
        cb = cb + 1 if wins_b[i] == 1 else 0
        sa.append(ca)
        sb.append(cb)
    m["streak_a"] = sa
    m["streak_b"] = sb
    m["flag"] = np.where(m["streak_a"] >= threshold, 1, np.where(m["streak_b"] >= threshold, -1, 0))
    return m[["date", "streak_a", "streak_b", "flag"]]

def five_day_return_diff(a_df, b_df, window=5):
    a = a_df[["date", "ret"]].rename(columns={"ret": "ret_a"})
    b = b_df[["date", "ret"]].rename(columns={"ret": "ret_b"})
    m = pd.merge(a, b, on="date", how="inner").sort_values("date")
    if m.empty:
        return pd.DataFrame(columns=["date", "diff", "roll"])
    m["ret_a"] = pd.to_numeric(m["ret_a"], errors="coerce")
    m["ret_b"] = pd.to_numeric(m["ret_b"], errors="coerce")
    m = m.dropna()
    m["diff"] = m["ret_a"] - m["ret_b"]
    m["roll"] = m["diff"].rolling(window=window, min_periods=1).sum()
    return m[["date", "diff", "roll"]]

def index_intraday_one_day(path, date_str):
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["time", "ret"])
    tcol = "time" if "time" in df.columns else ("datetime" if "datetime" in df.columns else df.columns[0])
    if tcol not in df.columns or "close" not in df.columns or "open" not in df.columns:
        return pd.DataFrame(columns=["time", "ret"])
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df["date"] = df[tcol].dt.strftime("%Y-%m-%d")
    sub = df[df["date"] == str(date_str)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["time", "ret"])
    sub = sub.sort_values(tcol)
    oc = pd.to_numeric(sub["open"], errors="coerce")
    cc = pd.to_numeric(sub["close"], errors="coerce")
    sub["ret"] = np.where(oc != 0, (cc / oc) - 1.0, np.nan)
    sub["time"] = sub[tcol].dt.strftime("%H:%M")
    sub = sub[["time", "ret"]].sort_values("time")
    return sub

def index_1m_dates():
    p300 = os.path.join("data", "indices_min1", "000300.XSHG_1m.csv")
    p1000 = os.path.join("data", "indices_min1", "000852.XSHG_1m.csv")
    dates = set()
    for p in [p300, p1000]:
        df = safe_read_csv(p)
        if df.empty:
            continue
        tcol = "time" if "time" in df.columns else ("datetime" if "datetime" in df.columns else df.columns[0])
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        ds = df[tcol].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
        for d in ds:
            dates.add(d)
    return sorted(list(dates))

def minute_win_streaks(df300, df1000):
    x = sorted(list(set(df300["time"].tolist() + df1000["time"].tolist())))
    m300 = {t: np.nan for t in x}
    m1000 = {t: np.nan for t in x}
    for _, r in df300.iterrows():
        m300[r["time"]] = r["ret"]
    for _, r in df1000.iterrows():
        m1000[r["time"]] = r["ret"]
    sa = []
    sb = []
    ca = 0
    cb = 0
    for t in x:
        ra = m300[t]
        rb = m1000[t]
        if pd.notna(ra) and pd.notna(rb):
            if ra > rb:
                ca += 1
                cb = 0
            elif rb > ra:
                cb += 1
                ca = 0
            else:
                ca = 0
                cb = 0
        else:
            ca = 0
            cb = 0
        sa.append(ca)
        sb.append(cb)
    return pd.DataFrame({"time": x, "streak300": sa, "streak1000": sb})

def intraday_index_streak_option(date_str, streak_df):
    x = streak_df["time"].tolist()
    s300 = streak_df["streak300"].tolist()
    s1000 = streak_df["streak1000"].tolist()
    series = [
        {"name": "HS300分钟连胜数", "type": "line", "smooth": True, "data": s300, "connectNulls": True},
        {"name": "中证1000分钟连胜数", "type": "line", "smooth": True, "data": s1000, "connectNulls": True},
    ]
    return line_option(x, series, f"{date_str} 指数1分钟连胜对比", "{value}")

def intraday_index_diff_option(date_str, df300, df1000):
    x = sorted(list(set(df300["time"].tolist() + df1000["time"].tolist())))
    m300 = {t: np.nan for t in x}
    m1000 = {t: np.nan for t in x}
    for _, r in df300.iterrows():
        m300[r["time"]] = r["ret"]
    for _, r in df1000.iterrows():
        m1000[r["time"]] = r["ret"]
    diff = []
    for t in x:
        ra = m300[t]
        rb = m1000[t]
        if pd.notna(ra) and pd.notna(rb):
            diff.append(round(float((ra - rb) * 100.0), 2))
        else:
            diff.append(None)
    series = [
        {
            "name": "分钟收益差(%)",
            "type": "line",
            "smooth": True,
            "data": diff,
            "connectNulls": True,
            "lineStyle": {"width": 2},
            "markLine": fixed_marklines([-0.1, -0.05, 0.05, 0.1], fmt="{c}%")
        }
    ]
    return line_option(x, series, f"{date_str} 指数1分钟收益差", "{value}%", ymin=-0.5, ymax=0.5, y_split_line_show=False)

def daily_index_three_option(title_text, df300, df1000):
    x = sorted(list(set(df300["date"].tolist() + df1000["date"].tolist())))
    m300 = {d: np.nan for d in x}
    m1000 = {d: np.nan for d in x}
    for _, r in df300.iterrows():
        m300[r["date"]] = r["ret"]
    for _, r in df1000.iterrows():
        m1000[r["date"]] = r["ret"]
    s300 = []
    s1000 = []
    c300 = 1.0
    c1000 = 1.0
    started300 = False
    started1000 = False
    for d in x:
        v300 = m300[d]
        v1000 = m1000[d]
        if pd.notna(v300):
            if not started300:
                s300.append(1.0)
                started300 = True
            else:
                c300 = c300 * (1.0 + float(v300) / 100.0)
                s300.append(round(c300, 4))
        else:
            s300.append(None)
        if pd.notna(v1000):
            if not started1000:
                s1000.append(1.0)
                started1000 = True
            else:
                c1000 = c1000 * (1.0 + float(v1000) / 100.0)
                s1000.append(round(c1000, 4))
        else:
            s1000.append(None)
 
    series = [
        {"name": "HS300累计收益", "type": "line", "smooth": True, "data": s300, "connectNulls": True, "yAxisIndex": 0},
        {"name": "中证1000累计收益", "type": "line", "smooth": True, "data": s1000, "connectNulls": True, "yAxisIndex": 0},
    ]
    return {
        "title": {"text": title_text, "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 40},
        "xAxis": {"type": "category", "data": x},
        "yAxis": [
            {"type": "value", "axisLabel": {"formatter": "{value}"}, "name": "累计基准=1"},
            {"type": "value", "axisLabel": {"formatter": "{value}%"}, "name": "差值(%)"}
        ],
        "grid": {"top": 80, "left": 60, "right": 40, "bottom": 50},
        "series": series
    }

def daily_group_ratio_spread_option(daily_df):
    x = daily_df["date"].tolist()
    la = pd.to_numeric(daily_df.get("large_ret", np.nan), errors="coerce")
    sm = pd.to_numeric(daily_df.get("small_ret", np.nan), errors="coerce")
    sp = pd.to_numeric(daily_df.get("spread", np.nan), errors="coerce")
    ratio = []
    for a, b in zip(la.tolist(), sm.tolist()):
        if pd.notna(a) and pd.notna(b):
            ra = 1.0 + float(a) / 100.0
            rb = 1.0 + float(b) / 100.0
            if rb != 0:
                ratio.append(round(ra / rb, 4))
            else:
                ratio.append(None)
        else:
            ratio.append(None)
    spread = [round(float(v), 2) if pd.notna(v) else None for v in sp.tolist()]
    series = [
        {"name": "收益比(大盘/小盘)", "type": "line", "smooth": True, "data": ratio, "connectNulls": True, "yAxisIndex": 0},
        {"name": "Spread(%)", "type": "line", "smooth": True, "data": spread, "connectNulls": True, "yAxisIndex": 1, "lineStyle": {"width": 2}}
    ]
    return {
        "title": {"text": "大小市值股票收益比与Spread对比", "left": "center", "top": 6},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 40},
        "xAxis": {"type": "category", "data": x},
        "yAxis": [
            {"type": "value", "axisLabel": {"formatter": "{value}"}, "name": "收益比(基准=1)"},
            {"type": "value", "axisLabel": {"formatter": "{value}%"}, "name": "Spread(%)"}
        ],
        "grid": {"top": 80, "left": 60, "right": 40, "bottom": 50},
        "series": series
    }

def daily_index_diff_spread_option(daily_df, idx300_df, idx1000_df):
    a = idx300_df[["date", "ret"]].rename(columns={"ret": "ret_a"})
    b = idx1000_df[["date", "ret"]].rename(columns={"ret": "ret_b"})
    m = pd.merge(a, b, on="date", how="inner").sort_values("date")
    m["ret_a"] = pd.to_numeric(m["ret_a"], errors="coerce")
    m["ret_b"] = pd.to_numeric(m["ret_b"], errors="coerce")
    m = m.dropna()
    m["idx_diff"] = m["ret_a"] - m["ret_b"]
    d = pd.merge(daily_df[["date", "spread"]], m[["date", "idx_diff"]], on="date", how="inner").sort_values("date")
    x = d["date"].tolist()
    idx_diff = [round(float(v), 2) for v in pd.to_numeric(d["idx_diff"], errors="coerce").tolist()]
    spr = [round(float(v), 2) for v in pd.to_numeric(d["spread"], errors="coerce").tolist()]
    series = [
        {"name": "指数日收益差(%)", "type": "line", "smooth": True, "data": idx_diff, "connectNulls": True, "lineStyle": {"width": 2}},
        {"name": "Spread(%)", "type": "line", "smooth": True, "data": spr, "connectNulls": True}
    ]
    return line_option(x, series, "指数日收益差与Spread对比", "{value}%")

def index_15m_one_day(path, date_str):
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["time", "ret"])
    tcol = "time" if "time" in df.columns else ("datetime" if "datetime" in df.columns else df.columns[0])
    if tcol not in df.columns or "close" not in df.columns or "open" not in df.columns:
        return pd.DataFrame(columns=["time", "ret"])
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df["date"] = df[tcol].dt.strftime("%Y-%m-%d")
    sub = df[df["date"] == str(date_str)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["time", "ret"])
    sub = sub.sort_values(tcol)
    oc = pd.to_numeric(sub["open"], errors="coerce")
    cc = pd.to_numeric(sub["close"], errors="coerce")
    hrs = sub[tcol].dt.hour
    mins = sub[tcol].dt.minute
    bucket_min = (mins // 15) * 15
    sub["bucket"] = hrs.map(lambda h: f"{h:02d}") + ":" + bucket_min.map(lambda m: f"{m:02d}")
    g = sub.groupby("bucket").agg({"open": "first", "close": "last"}).reset_index()
    g["ret"] = np.where(pd.to_numeric(g["open"], errors="coerce") != 0, (pd.to_numeric(g["close"], errors="coerce") / pd.to_numeric(g["open"], errors="coerce")) - 1.0, np.nan)
    g = g.rename(columns={"bucket": "time"})
    return g[["time", "ret"]].sort_values("time")

def intraday_spread_index15_option(date_str, small_df, large_df, idx_df):
    x = sorted(list(set(small_df["time"].tolist() + large_df["time"].tolist() + idx_df["time"].tolist())))
    s_map = {t: np.nan for t in x}
    l_map = {t: np.nan for t in x}
    i_map = {t: np.nan for t in x}
    for _, r in small_df.iterrows():
        s_map[r["time"]] = r["ret"]
    for _, r in large_df.iterrows():
        l_map[r["time"]] = r["ret"]
    for _, r in idx_df.iterrows():
        i_map[r["time"]] = r["ret"]
    sp = [round((float(l_map[t]) - float(s_map[t])) * 100.0, 2) if pd.notna(s_map[t]) and pd.notna(l_map[t]) else None for t in x]
    idx = [round(float(i_map[t]) * 100.0, 2) if pd.notna(i_map[t]) else None for t in x]
    series = [
        {"name": "15m Spread(%)", "type": "line", "smooth": True, "data": sp, "connectNulls": True, "lineStyle": {"width": 2}},
        {"name": "HS300 15m收益(%)", "type": "line", "smooth": True, "data": idx, "connectNulls": True}
    ]
    return line_option(x, series, f"{date_str} Spread与HS300指数15m对比", "{value}%")

def intraday_spread_index15diff_option(date_str, small_df, large_df, idx300_df, idx1000_df):
    x = sorted(list(set(small_df["time"].tolist() + large_df["time"].tolist() + idx300_df["time"].tolist() + idx1000_df["time"].tolist())))
    s_map = {t: np.nan for t in x}
    l_map = {t: np.nan for t in x}
    a_map = {t: np.nan for t in x}
    b_map = {t: np.nan for t in x}
    for _, r in small_df.iterrows():
        s_map[r["time"]] = r["ret"]
    for _, r in large_df.iterrows():
        l_map[r["time"]] = r["ret"]
    for _, r in idx300_df.iterrows():
        a_map[r["time"]] = r["ret"]
    for _, r in idx1000_df.iterrows():
        b_map[r["time"]] = r["ret"]
    sp = [round((float(l_map[t]) - float(s_map[t])) * 100.0, 2) if pd.notna(s_map[t]) and pd.notna(l_map[t]) else None for t in x]
    idiff = [round((float(a_map[t]) - float(b_map[t])) * 100.0, 2) if pd.notna(a_map[t]) and pd.notna(b_map[t]) else None for t in x]
    series = [
        {"name": "15m Spread(%)", "type": "line", "smooth": True, "data": sp, "connectNulls": True, "lineStyle": {"width": 2}},
        {"name": "指数15m收益差(%)", "type": "line", "smooth": True, "data": idiff, "connectNulls": True}
    ]
    return line_option(x, series, f"{date_str} Spread与指数15m收益差对比", "{value}%")

def main():
    st.set_page_config(page_title="大小风格研究可视化", layout="wide")
    st.title("大小风格研究可视化")
    boundaries = load_boundaries()
    daily = load_daily_group_returns()
    dates_daily = daily["date"].tolist() if not daily.empty else []
    dates_min = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join("data", "jq_output_min15", "small", "*.csv"))])
    dates_idx_1m = index_1m_dates()
    p300_1m = os.path.join("data", "indices_min1", "000300.XSHG_1m.csv")
    p1000_1m = os.path.join("data", "indices_min1", "000852.XSHG_1m.csv")
    idx300_d = index_daily_returns(os.path.join("data", "indices_day", "000300.XSHG_1d.csv"))
    idx1000_d = index_daily_returns(os.path.join("data", "indices_day", "000852.XSHG_1d.csv"))
    st.header("大小盘风格对应指数分析")
    if not idx300_d.empty and not idx1000_d.empty:
        st.subheader("1.1 双指数全年日线收益")
        opt_all = daily_index_three_option("HS300与中证1000 全年日线收益", idx300_d, idx1000_d)
        render_echarts(opt_all, 440)
        st.subheader("1.2 双指数收益差")
        ddf = five_day_return_diff(idx300_d, idx1000_d, window=5)
        if not ddf.empty:
            x3 = ddf["date"].tolist()
            s_daily = [round(float(v), 2) for v in ddf["diff"].tolist()]
            series3 = [
                {
                    "name": "当日收益差(%)",
                    "type": "line",
                    "smooth": True,
                    "data": s_daily,
                    "connectNulls": True,
                    "lineStyle": {"width": 2},
                    "markLine": fixed_marklines([-5, -0.5, 0.5, 3], fmt="{c}%")
                }
            ]
            opt3 = line_option(x3, series3, "HS300−中证1000 日收益差", "{value}%", ymin=-5, ymax=3, y_split_line_show=False)
            render_echarts(opt3, 420)
            thr = 0.5
            pos_cnt = int(np.sum([1 for v in s_daily if pd.notna(v) and float(v) > thr]))
            neg_cnt = int(np.sum([1 for v in s_daily if pd.notna(v) and float(v) < -thr]))
            st.caption(f">0.5%次数：{pos_cnt}；<-0.5%次数：{neg_cnt}")
    if dates_idx_1m:
        st.subheader("1.3 双指数一分钟收益差")
        sel_d = st.selectbox("选择日期(指数1分钟)", dates_idx_1m, index=max(0, len(dates_idx_1m) - 1))
        df300_rt = index_intraday_one_day(p300_1m, sel_d)
        df1000_rt = index_intraday_one_day(p1000_1m, sel_d)
        if not df300_rt.empty and not df1000_rt.empty:
            opt_min_diff = intraday_index_diff_option(sel_d, df300_rt, df1000_rt)
            render_echarts(opt_min_diff, 440)
            vals = [round(float((a - b) * 100.0), 2) if pd.notna(a) and pd.notna(b) else None for a, b in zip(df300_rt["ret"].tolist(), df1000_rt["ret"].tolist())]
            thr = 0.05
            pos_cnt = int(np.sum([1 for v in vals if v is not None and v > thr]))
            neg_cnt = int(np.sum([1 for v in vals if v is not None and v < -thr]))
            st.caption(f"分钟收益差统计：>0.05%次数 {pos_cnt}；<-0.05%次数 {neg_cnt}")
        else:
            st.info("该日指数分钟数据缺失")
    else:
        st.info("缺少指数1分钟数据")
    st.header("大小市值股票分析")
    st.subheader("2.1 大小市值界限图")
    if not boundaries.empty:
        x = boundaries["date"].tolist()
        series = [
            {"name": "大盘市值下界", "type": "line", "smooth": True, "showSymbol": False, "symbol": "none", "data": boundaries["large_lower"].tolist()},
            {"name": "小盘市值上界", "type": "line", "smooth": True, "showSymbol": False, "symbol": "none", "data": boundaries["small_upper"].tolist()}
        ]
        render_echarts(line_option(x, series, "大小市值界限折线图"), 420)
        st.caption("说明：展示每日市值分位阈值。large_lower 为进入大盘股的最低市值界限（前20%分位下界），small_upper 为进入小盘股的最高市值界限（后20%分位上界）。")
    else:
        st.info("缺少界限数据")
    st.subheader("2.2 每日Spread")
    if not daily.empty:
        x = daily["date"].tolist()
        y = pd.to_numeric(daily["spread"], errors="coerce").round(2).tolist()
        render_echarts(bar_option(x, y, "一年期每日Spread柱状图(正为大盘强，负为小盘强)"), 420)
        st.caption("说明：展示一年期每日风格差（Spread=大盘均值收益−小盘均值收益）。柱体为正表示当日大盘占优，为负表示当日小盘占优。单位为百分比。")
        st.subheader("2.3 15分钟平均收益")
        if dates_min:
            sel_d_bar = st.selectbox("选择日期(15分钟)", dates_min, index=max(0, len(dates_min) - 1), key="min_under_daily")
            s_b = load_intraday_group(sel_d_bar, "small")
            l_b = load_intraday_group(sel_d_bar, "large")
            if not s_b.empty and not l_b.empty:
                render_echarts(intraday_option(sel_d_bar, s_b, l_b), 420)
            else:
                st.info("该日分钟数据缺失")
        else:
            st.info("缺少分钟数据")
        st.subheader("2.4 近三个月15m平均Spread")
        prof = minute_profile_last_month()
        if not prof.empty:
            opt = line_option(prof["time"].tolist(), [{"name": "15分钟均Spread(%)", "type": "line", "smooth": True, "data": (prof["spread"] * 100.0).round(2).tolist()}], "近三个月按分钟平均Spread", "{value}%")
            render_echarts(opt, 420)
        st.subheader("2.5 直方图")
        hist = pd.to_numeric(daily["spread"], errors="coerce").dropna().tolist()
        edges = np.arange(-3.5, 6.5 + 0.500001, 0.5)
        counts, edges = np.histogram(hist, bins=edges)
        labels = [f"[{round(edges[i], 2)},{round(edges[i + 1], 2)})" for i in range(len(edges) - 1)]
        thr_labels = []
        render_echarts(hist_option(labels, counts.tolist(), "Spread分布直方图", vlines=thr_labels), 420)
        st.caption("说明：展示一年期每日 Spread 的统计分布；横轴为区间范围，固定范围[-3.5, 6.5]，间隔0.5；纵轴为该区间内的频数。")
    else:
        st.info("缺少每日涨跌幅数据")
    st.header("大小市值股票Spread与指数涨跌综合分析")
    if not daily.empty:
        st.subheader("3.1 HS300散点图")
        idx300 = index_daily_returns(os.path.join("data", "indices_day", "000300.XSHG_1d.csv"))
        if not idx300.empty:
            merged = pd.merge(daily, idx300, on="date", how="inner")
            x_s = pd.to_numeric(merged["ret"], errors="coerce").tolist()
            y_s = pd.to_numeric(merged["spread"], errors="coerce").tolist()
            render_echarts(scatter_option(x_s, y_s, "Spread与沪深300日收益相关", "HS300日收益(%)", "Spread(%)"), 420)
            q = quadrant_ratios(x_s, y_s)
            st.caption(f"象限占比：右上{q['q1']}%，左上{q['q2']}%，左下{q['q3']}%，右下{q['q4']}%")
        st.subheader("3.2 中证1000散点图")
        idx1000 = index_daily_returns(os.path.join("data", "indices_day", "000852.XSHG_1d.csv"))
        if not idx1000.empty:
            merged2 = pd.merge(daily, idx1000, on="date", how="inner")
            x_2 = pd.to_numeric(merged2["ret"], errors="coerce").tolist()
            y_2 = pd.to_numeric(merged2["spread"], errors="coerce").tolist()
            render_echarts(scatter_option(x_2, y_2, "Spread与中证1000日收益相关", "中证1000日收益(%)", "Spread(%)"), 420)
            q2 = quadrant_ratios(x_2, y_2)
            st.caption(f"象限占比：右上{q2['q1']}%，左上{q2['q2']}%，左下{q2['q3']}%，右下{q2['q4']}%")
        if not idx300.empty and not idx1000.empty:
            st.subheader("3.3 全年日线收益差值与Spread对比")
            opt_is = daily_index_diff_spread_option(daily, idx300, idx1000)
            render_echarts(opt_is, 420)
        st.subheader("3.4 每日15m线收益差值与Spread对比")
        if dates_min:
            sel_d_cmp = st.selectbox("选择日期(15分钟对比)", dates_min, index=max(0, len(dates_min) - 1), key="min_for_index_spread_compare")
            s_b2 = load_intraday_group(sel_d_cmp, "small")
            l_b2 = load_intraday_group(sel_d_cmp, "large")
            p300_1m_loc = os.path.join("data", "indices_min1", "000300.XSHG_1m.csv")
            p1000_1m_loc = os.path.join("data", "indices_min1", "000852.XSHG_1m.csv")
            idx15_a = index_15m_one_day(p300_1m_loc, sel_d_cmp)
            idx15_b = index_15m_one_day(p1000_1m_loc, sel_d_cmp)
            if not s_b2.empty and not l_b2.empty and not idx15_a.empty and not idx15_b.empty:
                opt_si15d = intraday_spread_index15diff_option(sel_d_cmp, s_b2, l_b2, idx15_a, idx15_b)
                render_echarts(opt_si15d, 420)
            else:
                st.info("该日分钟或指数数据缺失")
        else:
            st.info("缺少分钟数据")

if __name__ == "__main__":
    main()
