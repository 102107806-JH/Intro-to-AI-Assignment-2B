import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime
from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
import datetime as dt
import numpy as np
from jh_ml_models.model_deployment_abstractions.deployment_data_testing.deployment_data_model_tester import DeploymentDataModelTester


nodes_df = pd.read_excel("./data/graph_init_data.xlsx")  # Load SCATS node metadata
nodes_df.rename(  # Renames columns
    columns={
        "SCATS Number": "SCATS_Number",
        "NB_LATITUDE": "lat",
        "NB_LONGITUDE": "lon",
    },
    inplace=True,
)

gdf = gpd.read_file("./GUI/vic_lga.shp") # Load map boundaries
boroondara = gdf[gdf["LGA_NAME"].str.contains("Boroondara", case=False)]  # Filter the Victoria data to only Boroondara
boroondara_geojson = boroondara.__geo_interface__
model_df = pd.read_excel("./data/data_base.xlsx", sheet_name="Current_Data")  # Load traffic data for timeline (hourly)
model_df["Date"] = pd.to_datetime(model_df["Date"], dayfirst=True)  # Convert date to datetime while loading the Excel column
time_cols = [c for c in model_df.columns if isinstance(c, dt.time)]  # Get the time columns
id_cols = [c for c in model_df.columns if c not in time_cols]  # Get the id columns

long_df = model_df.melt(  # Convert to long format
    id_vars=id_cols, value_vars=time_cols,
    var_name="tod", value_name="count"
)
long_df["datetime"] = pd.to_datetime(  # Convert datetime to string
    long_df["Date"].dt.date.astype(str) + " " + long_df["tod"].astype(str)
)
long_df["date"] = long_df["datetime"].dt.date  # Get date from datetime
long_df["hour_ts"] = long_df["datetime"].dt.floor("h")  # Get hour from datetime

hourly = (  # Hourly data
    long_df.groupby(["SCATS_Number", "date", "hour_ts"], as_index=False)["count"]
    .sum()
    .rename(columns={"count": "volume"})
    .sort_values(["SCATS_Number", "date", "hour_ts"])
)
hourly["prev_volume"] = hourly.groupby(["SCATS_Number", "date"])["volume"].shift(1)
hourly["delta"] = hourly["volume"] - hourly["prev_volume"]
hourly["pct_change"] = (hourly["delta"] / hourly["prev_volume"].replace(0, np.nan)) * 100

daily_extrema = ( # Peak/low daily markers
    hourly.groupby(["SCATS_Number", "date"])
    .agg(peak_volume=("volume", "max"), low_volume=("volume", "min"))
    .reset_index()
)
hourly = hourly.merge(daily_extrema, on=["SCATS_Number", "date"], how="left")
hourly["is_peak"] = hourly["volume"] == hourly["peak_volume"]
hourly["is_low"] = hourly["volume"] == hourly["low_volume"]
traffic_df = hourly.merge(nodes_df, on="SCATS_Number", how="inner")


def compute_color(row):
    """"
    Set the color for each node based on peak/low/no data
    :param row: pandas dataframe row
    :return: color for the node
    """
    if row["is_peak"]:
        return "white"
    if row["is_low"]:
        return "gray"
    if pd.notna(row["delta"]):
        if row["delta"] > 0:
            return "#FFD21F"  # yellow
        if row["delta"] < 0:
            return "#1F77FF"  # blue
    return "#BDBDBD"  # flat/no data


def base_figure():
    """
    Create the map figure for the Dash GUI by loading the geojson file and using the mapbox library to create the map.
    :param boroondara_geojson:
    :return: figure
    """
    fig = go.Figure()
    fig.update_layout(
        mapbox_style="mapbox://styles/mapbox/satellite-streets-v12",
        mapbox_accesstoken="pk.eyJ1IjoiczEwNTMzNDEyOCIsImEiOiJjbWYwdTZqNXcwczc0MmpvZmJ0N2Z4OHN2In0.oxNCZNuCeBQ0BBK_UDqZ3g",
        mapbox=dict(
            center=dict(lat=-37.8303, lon=145.0477),
            zoom=12.3,
            layers=[
                {
                    "sourcetype": "geojson",
                    "source": boroondara_geojson,
                    "type": "line",
                    "color": "red",
                    "line": {"width": 3},
                }
            ],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


app = dash.Dash(__name__)
available_dates = sorted(traffic_df["date"].unique()) if len(traffic_df) else []
default_date = available_dates[-1] if available_dates else dt.date.today()
app.layout = html.Div(
    [
        html.H1("Boroondara SCATS Path Finder"),
        html.Div(
            [
                html.Label("Origin SCATS Site"),
                dcc.Dropdown(
                    id="origin-input",
                    options=[{"label": str(i), "value": i} for i in nodes_df["SCATS_Number"]],
                    value=970,
                ),
                html.Label("Destination SCATS Site"),
                dcc.Dropdown(
                    id="destination-input",
                    options=[{"label": str(i), "value": i} for i in nodes_df["SCATS_Number"]],
                    value=2000,
                ),
                html.Label("Model Type"),
                dcc.Dropdown(
                    id="model-type",
                    options=[
                        {"label": "LSTM", "value": "LSTM"},
                        {"label": "GRU", "value": "GRU"},
                        {"label": "TCN", "value": "TCN"},
                    ],
                    value="LSTM",
                ),
                html.Label("Sequence Length"),
                dcc.Input(id="sequence-length", type="number", value=12, min=1, disabled=True),
                # New Div for K Value to put them on the same line
                html.Div([
                    html.Label("K Value:"),
                    dcc.Input(id="k-val", type="number", value=1, min=1, style={'marginLeft': '5px'})
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Button("Find Path", id="submit-button", n_clicks=0),
                html.Hr(),
                html.Label("Date"),
                dcc.Dropdown(
                    id="date-dropdown",
                    options=[{"label": str(d), "value": str(d)} for d in available_dates],
                    value=str(default_date),
                    clearable=False,
                ),
                html.Label("Hour of Day"),
                dcc.Slider(
                    id="hour-slider",
                    min=0, max=23, step=1, value=8,
                    marks={h: f"{h:02d}:00" for h in range(0, 24, 2)},
                ),
                html.Div(
                    [
                        html.Button("⏵ Play", id="play-btn", n_clicks=0, style={"marginRight": "8px"}),
                        html.Button("⏸ Pause", id="pause-btn", n_clicks=0),
                        html.Span(id="hour-readout", style={"marginLeft": "12px", "fontWeight": "bold"}),
                    ],
                    style={"marginTop": "10px"},
                ),
                dcc.Interval(id="timer", interval=1200, n_intervals=0, disabled=True),
                dcc.Graph(id="tester-graph", figure=go.Figure(), style={"height": "300px", "marginTop": "20px"}),
            ],
            style={"width": "30%", "display": "inline-block", "verticalAlign": "top"},
        ),
        html.Div(
            [dcc.Graph(id="map", figure=base_figure())],
            style={"marginTop": "10px", "width": "65%", "display": "inline-block"},
        ),
    ]
)


@app.callback(
    [Output("map", "figure"), Output("tester-graph", "figure"), Output("hour-readout", "children")],
    Input("submit-button", "n_clicks"),
    Input("date-dropdown", "value"),
    Input("hour-slider", "value"),
    State("origin-input", "value"),
    State("destination-input", "value"),
    State("model-type", "value"),
    State("sequence-length", "value"),
    State("k-val", "value"),
)
def update_map(n_clicks, date_val, hour_val, origin, destination, model_type, sequence_length, k_val):
    map_fig = base_figure()
    graph_builder = GraphVertexEdgeInit("./GUI/graph_init_data.xlsx")
    graph = graph_builder.extract_file_contents()

    if date_val:
        date_val = pd.to_datetime(date_val).date()
        hour_ts = pd.Timestamp.combine(date_val, dt.time(hour_val, 0))
        subset = traffic_df[traffic_df["hour_ts"] == hour_ts].copy()
        if not subset.empty:
            subset["color"] = subset.apply(compute_color, axis=1)
            hover_text = (
                    "SCATS: " + subset["SCATS_Number"].astype(str) +
                    "<br>Hour: " + subset["hour_ts"].astype(str) +
                    "<br>Volume: " + subset["volume"].astype(int).astype(str) +
                    "<br>Δ cars: " + subset["delta"].fillna(0).astype(int).astype(str) +
                    "<br>% change: " + subset["pct_change"].fillna(0).round(1).astype(str) + "%"
            )
            map_fig.add_trace(
                go.Scattermapbox(
                    lat=subset["lat"],
                    lon=subset["lon"],
                    mode="markers",
                    marker=dict(size=12, color=subset["color"], opacity=0.9),
                    text=hover_text,
                    hoverinfo="text",
                    name="Traffic Volume",
                )
            )

    if n_clicks > 0:
        current_time = datetime.now().replace(month=8)
        path_finder = PathFinder(graph=graph)
        solution_nodes = path_finder.find_paths(
            initial_state=origin,
            goal_state=destination,
            initial_time=current_time,
            sequence_length=sequence_length,
            k_val=k_val,
            mode=model_type,
        )
        if solution_nodes:
            nodes_iter = solution_nodes if hasattr(solution_nodes, "__len__") else [solution_nodes]
            colors = ["purple", "green", "orange", "cyan", "yellow", "pink", "brown"]
            for i, path in enumerate(nodes_iter):
                path_states, node = [], path
                while hasattr(node, "state") and node:
                    path_states.append(node.state)
                    node = getattr(node, "parent", None)
                path_states.reverse()
                if path_states:
                    coords = nodes_df.set_index("SCATS_Number").loc[path_states]
                    total_time_minutes = path.time_cost
                    total_time_hours = total_time_minutes
                    time_text = f"Est. Time: {total_time_hours:.2f} hours"
                    map_fig.add_trace(
                        go.Scattermapbox(
                            lat=coords["lat"],
                            lon=coords["lon"],
                            mode="lines+markers",
                            line=dict(width=4, color=colors[i % len(colors)]),
                            marker=dict(size=10, color=colors[i % len(colors)]),
                            text=[str(s) for s in path_states],
                            name="Optimal Path ({})".format(time_text) if i == 0 else "Path {} ({})".format(i+1, time_text),
                        )
                    )

    start_datetime = datetime(year=2025, month=8, day=1, hour=0, minute=0)
    end_datetime = datetime(year=2025, month=8, day=2, hour=0, minute=0)
    model_tester = DeploymentDataModelTester(database_file_path="data/data_base.xlsx")
    results = model_tester.test_models(
        scats_site=origin,
        prediction_depth=1,
        sequence_length=sequence_length,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    targets = np.asarray(results.get("Targets", []), dtype=float)
    gru_pred = np.asarray(results.get("GRU", np.zeros_like(targets)), dtype=float)
    tcn_pred = np.asarray(results.get("TCN", np.zeros_like(targets)), dtype=float)
    lstm_pred = np.asarray(results.get("LSTM", np.zeros_like(targets)), dtype=float)
    tester_fig = go.Figure()
    if len(targets) > 0:
        abs_gru = np.abs(targets - gru_pred)
        abs_tcn = np.abs(targets - tcn_pred)
        abs_lstm = np.abs(targets - lstm_pred)
        avg_gru = np.nan if len(abs_gru) == 0 else float(np.mean(abs_gru))
        avg_tcn = np.nan if len(abs_tcn) == 0 else float(np.mean(abs_tcn))
        avg_lstm = np.nan if len(abs_lstm) == 0 else float(np.mean(abs_lstm))
        tester_fig.add_trace(go.Scatter(
            y=abs_gru, mode="lines",
            name=f"GRU ABS (avg {avg_gru:.2f})",
            line=dict(color="blue")
        ))
        tester_fig.add_trace(go.Scatter(
            y=abs_tcn, mode="lines",
            name=f"TCN ABS (avg {avg_tcn:.2f})",
            line=dict(color="orange")
        ))
        tester_fig.add_trace(go.Scatter(
            y=abs_lstm, mode="lines",
            name=f"LSTM ABS (avg {avg_lstm:.2f})",
            line=dict(color="red")
        ))
        tester_fig.update_layout(
            title=f"Per-timestep Absolute Error (SCATS {origin})",
            xaxis_title="Timestep index",
            yaxis_title="Absolute error (TFV)",
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        tester_fig.update_layout(title="No test data available for selected site/time")
    return map_fig, tester_fig, f"Showing {str(date_val)} at {hour_val:02d}:00"


@app.callback(
    Output("timer", "disabled"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    prevent_initial_call=True
)
def play_pause(n_play, n_pause):
    """
    Creates a timer that can be used to control the play/pause functionality of the map.
    :param n_play: Number of clicks on the play button.
    :return True if the play button was clicked, False if the pause button was clicked.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    return False if trig == "play-btn" else True


@app.callback(
    Output("hour-slider", "value"),
    Input("timer", "n_intervals"),
    State("hour-slider", "value"),
    prevent_initial_call=True
)
def tick(_n, hour_val):
    """
    Creates the slider for the hour value and adds a callback to update the hour value.
    :param _n: The number of ticks.
    :param hour_val: The current hour value.
    :return: The next hour value to update
    """
    return 0 if hour_val >= 23 else hour_val + 1

if __name__ == "__main__":
    app.run(debug=True)  # Set to True for debugging, otherwise False