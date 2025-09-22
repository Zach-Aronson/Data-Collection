# app_force_viewer.py
# Force vs Time viewer with live per-run averages and Crr (mass in lb, speed parsed from filename).
# - Table spans the bottom (full width)
# - Title moved outside the plot to prevent overlap with legend
# - Export stats CSV button
# - Crr_basic = mean(F) / (m*g)
# - Crr_aero  = mean(F - 0.5*rho*CdA*v^2) / (m*g) using each run's parsed speed
# - Choose CdA directly OR Cd × Area (Area default 0.9 m²)
# - Level surface, no wind, rho = 1.225 kg/m³

import base64, io, os, re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import os

from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme  # Capital F + valid Scheme

GRAVITY     = 9.80665       # m/s^2
AIR_DENSITY = 1.225         # kg/m^3 (ISA sea-level)
LB_TO_KG    = 0.45359237

# ---------- Helpers ----------
def _find_col(df: pd.DataFrame, candidates: List[str]):
    norm = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    for key, orig in norm.items():
        for cand in candidates:
            if cand in key:
                return orig
    return None

def _to_force_N(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    return (s / 1000.0) * GRAVITY if s.max() > 100 else s

def _to_time_s(series: pd.Series) -> pd.Series:
    t = pd.to_numeric(series, errors="coerce")
    if t.dropna().empty:
        return t
    return t / 1000.0 if t.max() > 50 else t

def parse_speed_from_name(name: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r'(\d+(?:\.\d+)?)\s*mph', name, flags=re.IGNORECASE)
    if not m:
        return None, None
    v_mph = float(m.group(1))
    v_mps = v_mph * 0.44704
    return v_mps, v_mph

def parse_csv_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    sio = io.StringIO(content_bytes.decode("utf-8", errors="ignore"))
    df = pd.read_csv(sio, sep=None, engine="python")
    tcol = _find_col(df, ["time(ms)", "time_ms", "timems", "time(s)", "times", "time"])
    ycol = _find_col(df, ["weight(g)", "weightg", "mass(g)", "massg",
                          "weight(kg)", "mass(kg)", "weight", "mass", "n", "newton", "force"])
    if tcol is None or ycol is None:
        raise ValueError(f"{os.path.basename(filename)}: could not detect time/weight columns (found {list(df.columns)})")
    time_s = _to_time_s(df[tcol])
    force_n = _to_force_N(df[ycol])
    base = os.path.basename(filename)
    v_mps, v_mph = parse_speed_from_name(base)
    return pd.DataFrame({
        "file": base,
        "time_s": time_s,
        "force_N": force_n,
        "speed_mps": v_mps,
        "speed_mph": v_mph
    }).dropna(subset=["time_s", "force_N"])

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    return parse_csv_bytes(raw, filename)

def which_triggered():
    if not callback_context.triggered:
        return None
    return callback_context.triggered[0]["prop_id"].split(".")[0]

def extract_xrange_from_relayout(relayout: Optional[dict]):
    if not isinstance(relayout, dict):
        return None, None, None
    return (relayout.get("xaxis.range[0]"),
            relayout.get("xaxis.range[1]"),
            relayout.get("xaxis.autorange"))

def extract_xrange_from_figure(fig: dict):
    try:
        rng = fig["layout"]["xaxis"].get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            return rng[0], rng[1]
    except Exception:
        pass
    return None, None

# ---------- App ----------
app = Dash(__name__)
app.title = "Force vs Time Viewer"

fmt3 = Format(precision=3, scheme=Scheme.fixed, group=False)
fmt4 = Format(precision=4, scheme=Scheme.fixed, group=False)

app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "padding": "14px"},
    children=[
        html.H2("Force vs Time (interactive)"),  # title OUTSIDE the plot
        # Controls row
        html.Div(
            [
                html.Label("Vehicle mass (lb)"),
                dcc.Input(id="mass_lb", type="number", value=3300, min=1, step=1,
                          style={"width": "110px", "marginRight": "16px"}),

                html.Label("Direct CdA (m²)"),
                dcc.Input(id="cda_direct", type="number", value=0.0, min=0, step=0.01,
                          style={"width": "100px", "marginRight": "16px"}),

                html.Label("Cd"),
                dcc.Input(id="cd", type="number", value=0.30, min=0, step=0.01,
                          style={"width": "80px", "marginRight": "10px"}),

                html.Label("Area A (m²)"),
                dcc.Input(id="area", type="number", value=0.9, min=0, step=0.01,
                          style={"width": "100px", "marginRight": "16px"}),

                html.Span("Assumptions: per-run speed from filename, level road, no wind, ρ = 1.225 kg/m³",
                          style={"color": "#666"}),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "6px",
                   "flexWrap": "wrap", "marginBottom": "10px"}
        ),

        # Uploader + file list
        dcc.Upload(
            id="uploader",
            children=html.Div(["Drag & Drop or ", html.A("Click to Select Files"), " (.txt, .csv, .log)"]),
            multiple=True,
            style={
                "width": "100%", "height": "80px", "lineHeight": "80px",
                "borderWidth": "2px", "borderStyle": "dashed",
                "borderRadius": "8px", "textAlign": "center", "marginBottom": "10px"
            },
        ),
        html.Div(id="filelist", style={"marginBottom": "8px", "color": "#888"}),

        # Plot + Y controls
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="graph", style={"height": "62vh"}, config={"displaylogo": False}),
                        html.Div(id="x_hint", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"}),
                    ],
                    style={"flex": "1 1 auto", "minWidth": "0"}
                ),
                html.Div(
                    [
                        html.Div("Y Range", style={"fontWeight": 600, "marginBottom": "6px"}),
                        dcc.RangeSlider(
                            id="y_range", min=0, max=1, step=None, value=[0, 1],
                            vertical=True, tooltip={"placement": "left", "always_visible": True}, allowCross=False,
                        ),
                        html.Div(id="y_range_labels", style={"marginTop": "6px", "fontSize": "12px", "color": "#666"}),
                        html.Button("Auto (All)", id="btn_yauto", n_clicks=0, style={"marginTop": "10px", "width": "100%"}),
                        html.Button("Central 98%", id="btn_y98", n_clicks=0, style={"marginTop": "6px", "width": "100%"}),
                        html.Button("Central 95%", id="btn_y95", n_clicks=0, style={"marginTop": "6px", "width": "100%"}),
                    ],
                    style={"width": "140px", "padding": "0 0 0 12px"}
                )
            ],
            style={"display": "flex", "alignItems": "stretch", "marginBottom": "10px"}
        ),

        # Bottom: controls + table + download
        html.Div(
            [
                html.Div(
                    [
                        html.Div(id="cda_effective",
                                 style={"fontSize": "12px", "color": "#444", "margin": "0 0 6px 2px"}),
                        html.Button("Export stats CSV", id="btn_export", n_clicks=0,
                                    style={"margin": "0 0 8px 0"}),
                        dcc.Download(id="download_stats"),
                        DataTable(
                            id="stats_table",
                            columns=[
                                {"name": "Run",                "id": "file"},
                                {"name": "Speed (mph)",        "id": "speed_mph", "type": "numeric", "format": fmt3},
                                {"name": "Avg Force (N)",      "id": "avg_force", "type": "numeric", "format": fmt3},
                                {"name": "Aero Drag (N)",      "id": "f_aero",    "type": "numeric", "format": fmt3},
                                {"name": "Rolling Force (N)",  "id": "f_rolling", "type": "numeric", "format": fmt3},
                                {"name": "Crr (basic)",        "id": "crr_basic", "type": "numeric", "format": fmt4},
                                {"name": "Crr (aero)",         "id": "crr_aero",  "type": "numeric", "format": fmt4},
                                {"name": "Points",             "id": "count",     "type": "numeric"},
                            ],
                            data=[],
                            sort_action="native",
                            style_table={
                                "width": "100%",
                                "overflowX": "auto",
                                "maxHeight": "26vh",
                                "overflowY": "auto",
                                "border": "1px solid #eee",
                            },
                            style_cell={"fontSize": "12px", "padding": "6px 8px"},
                            style_header={"fontWeight": "600"},
                            style_as_list_view=True,
                        ),
                    ],
                    style={"width": "100%"}
                ),
            ],
            style={"width": "100%"}
        ),

        dcc.Store(id="data_store"),
        dcc.Store(id="y_stats_store"),
    ]
)

# ---------- Callbacks ----------
@app.callback(
    Output("data_store", "data"),
    Output("filelist", "children"),
    Input("uploader", "contents"),
    State("uploader", "filename"),
    prevent_initial_call=True
)
def load_files(contents_list, names_list):
    if not contents_list:
        return no_update, ""
    dfs, skipped = [], []
    for contents, name in zip(contents_list, names_list):
        try:
            dfs.append(parse_upload(contents, name))
        except Exception as e:
            skipped.append(f"{name}: {e}")
    if not dfs:
        return None, html.Div("No valid files parsed.", style={"color": "crimson"})
    data = pd.concat(dfs, ignore_index=True)

    files = sorted(set(data["file"]))
    speeds_info = []
    for f in files:
        v = data.loc[data["file"] == f, "speed_mph"].dropna()
        if v.empty:
            speeds_info.append(f"{f}: speed not found (expected '##mph')")
    msg = "Loaded runs: " + ", ".join(files)
    if speeds_info:
        msg += " | " + " ; ".join(speeds_info)
    return data.to_dict("records"), msg

@app.callback(
    Output("y_stats_store", "data"),
    Output("y_range", "min"),
    Output("y_range", "max"),
    Output("y_range", "value"),
    Output("y_range_labels", "children"),
    Input("data_store", "data"),
    prevent_initial_call=True
)
def init_y_slider(records):
    if not records:
        return None, 0, 1, [0, 1], ""
    df = pd.DataFrame(records)
    ymin = float(df["force_N"].min()); ymax = float(df["force_N"].max())
    span = max(ymax - ymin, 1e-9); pad = 0.02 * span
    q02, q98 = df["force_N"].quantile([0.02, 0.98])
    q025, q975 = df["force_N"].quantile([0.025, 0.975])
    stats = {"ymin": ymin, "ymax": ymax, "q02": float(q02), "q98": float(q98),
             "q025": float(q025), "q975": float(q975)}
    labels = f"Min: {ymin:.3f}  |  Max: {ymax:.3f}"
    return stats, ymin - pad, ymax + pad, [ymin, ymax], labels

@app.callback(
    Output("graph", "figure"),
    Output("x_hint", "children"),
    Input("data_store", "data"),
    Input("y_range", "value"),
    Input("btn_yauto", "n_clicks"),
    Input("btn_y98", "n_clicks"),
    Input("btn_y95", "n_clicks"),
    Input("graph", "relayoutData"),
    State("y_stats_store", "data"),
)
def update_graph(records, y_range, n_auto, n_98, n_95, relayout, ystats):
    fig = make_subplots(rows=1, cols=1)

    if records:
        df = pd.DataFrame(records)
        for fname, grp in df.groupby("file", sort=False):
            grp = grp.sort_values("time_s")
            fig.add_trace(go.Scattergl(
                x=grp["time_s"], y=grp["force_N"],
                mode="lines+markers", marker=dict(size=3), line=dict(width=1),
                name=fname,
                hovertemplate="File: %{fullData.name}<br>t=%{x:.3f}s<br>F=%{y:.3f}N<extra></extra>",
            ))

    # Keep title OFF here; place legend above and add headroom
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.10,   # high enough to clear anything
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=60, r=20, t=120, b=90),  # generous top for legend
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        uirevision="keep-zoom",
    )

    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1s", step="second", stepmode="backward"),
            dict(count=2, label="2s", step="second", stepmode="backward"),
            dict(count=5, label="5s", step="second", stepmode="backward"),
            dict(step="all"),
        ])
    )

    xmin, xmax, x_auto = extract_xrange_from_relayout(relayout)
    x_hint = ""
    if x_auto is True:
        fig.update_xaxes(autorange=True, range=None)
        x_hint = "X range: auto"
    elif xmin is not None and xmax is not None:
        fig.update_xaxes(autorange=False, range=[xmin, xmax])
        x_hint = f"X range: {float(xmin):.3f}s → {float(xmax):.3f}s"

    trig = which_triggered()
    if ystats:
        if trig == "btn_yauto":
            fig.update_yaxes(autorange=True)
        elif trig == "btn_y98":
            fig.update_yaxes(range=[ystats["q02"], ystats["q98"]], autorange=False)
        elif trig == "btn_y95":
            fig.update_yaxes(range=[ystats["q025"], ystats["q975"]], autorange=False)
        else:
            fig.update_yaxes(range=y_range, autorange=False)

    return fig, x_hint

# ---- stats computation (shared) ----
def compute_stats(records, relayout, figure, mass_lb, cda_direct, cd, area):
    if not records:
        return "", pd.DataFrame(columns=["file","speed_mph","avg_force","f_aero","f_rolling","crr_basic","crr_aero","count"])
    df = pd.DataFrame(records)

    xmin, xmax, x_auto = extract_xrange_from_relayout(relayout)
    if (xmin is None or xmax is None) and isinstance(figure, dict):
        fxmin, fxmax = extract_xrange_from_figure(figure)
        if fxmin is not None and fxmax is not None:
            xmin, xmax = fxmin, fxmax
            x_auto = False

    if xmin is not None and xmax is not None and (x_auto is False or x_auto is None):
        mask = (df["time_s"] >= float(xmin)) & (df["time_s"] <= float(xmax))
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask]
    files = sorted(df["file"].unique())

    # mass in kg
    m = float(mass_lb) * LB_TO_KG if (mass_lb is not None and mass_lb > 0) else None

    # effective CdA
    if cd is not None and cd > 0 and area is not None and area > 0:
        cda_val = float(cd) * float(area)
        cda_text = f"Using Cd × A = {cd:.3f} × {area:.3f} → CdA = {cda_val:.3f} m²"
    else:
        cda_val = float(cda_direct) if (cda_direct is not None and cda_direct >= 0) else 0.0
        cda_text = f"Using direct CdA = {cda_val:.3f} m²"

    if sub.empty:
        out = pd.DataFrame({
            "file": files,
            "speed_mph": [np.nan]*len(files),
            "avg_force": [np.nan]*len(files),
            "f_aero": [np.nan]*len(files),
            "f_rolling": [np.nan]*len(files),
            "crr_basic": [np.nan]*len(files),
            "crr_aero": [np.nan]*len(files),
            "count": [0]*len(files),
        })
        return cda_text, out

    stats = (
        sub.groupby("file", sort=False)
           .agg(avg_force=("force_N", "mean"),
                count=("force_N", "size"),
                speed_mph=("speed_mph", "first"),
                speed_mps=("speed_mps", "first"))
           .reset_index()
    )

    v2 = (stats["speed_mps"].fillna(0.0)) ** 2
    F_aero = 0.5 * AIR_DENSITY * cda_val * v2
    stats["f_aero"] = F_aero
    stats["f_rolling"] = stats["avg_force"] - stats["f_aero"]

    if m is not None and m > 0:
        stats["crr_basic"] = stats["avg_force"] / (m * GRAVITY)
        stats["crr_aero"]  = stats["f_rolling"] / (m * GRAVITY)
    else:
        stats["crr_basic"] = np.nan
        stats["crr_aero"]  = np.nan

    all_files = pd.DataFrame({"file": files})
    out = all_files.merge(stats, on="file", how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out = out[["file", "speed_mph", "avg_force", "f_aero", "f_rolling", "crr_basic", "crr_aero", "count"]]
    return cda_text, out

@app.callback(
    Output("cda_effective", "children"),
    Output("stats_table", "data"),
    Input("data_store", "data"),
    Input("graph", "relayoutData"),
    Input("graph", "figure"),
    Input("mass_lb", "value"),
    Input("cda_direct", "value"),
    Input("cd", "value"),
    Input("area", "value"),
    prevent_initial_call=True
)
def update_stats(records, relayout, figure, mass_lb, cda_direct, cd, area):
    cda_text, df_stats = compute_stats(records, relayout, figure, mass_lb, cda_direct, cd, area)
    return cda_text, df_stats.to_dict("records")

# ---- download CSV ----
@app.callback(
    Output("download_stats", "data"),
    Input("btn_export", "n_clicks"),
    State("data_store", "data"),
    State("graph", "relayoutData"),
    State("graph", "figure"),
    State("mass_lb", "value"),
    State("cda_direct", "value"),
    State("cd", "value"),
    State("area", "value"),
    prevent_initial_call=True
)
def export_stats(n_clicks, records, relayout, figure, mass_lb, cda_direct, cd, area):
    if not n_clicks:
        return no_update
    cda_text, df_stats = compute_stats(records, relayout, figure, mass_lb, cda_direct, cd, area)
    # Include the CdA note as CSV metadata by adding a row at top:
    df_out = df_stats.copy()
    # Return CSV
    return dcc.send_data_frame(df_out.to_csv, "force_stats.csv", index=False)

# expose WSGI app for Gunicorn
server = app.server

if __name__ == "__main__":
    # local/dev run; Render uses gunicorn with `server` above
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

