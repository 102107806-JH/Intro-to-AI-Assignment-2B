import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime
from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit

nodes_df = pd.read_excel("./data/graph_init_data.xlsx")  # Load SCATS node coordinates
nodes_df.rename(
    columns={
        "SCATS Number": "SCATS_Number",
        "NB_LATITUDE": "lat",
        "NB_LONGITUDE": "lon",
    },
    inplace=True,
)

gdf = gpd.read_file("./GUI/vic_lga.shp")
boroondara = gdf[gdf["LGA_NAME"].str.contains("Boroondara", case=False)]  # Load Boroondara boundary
boroondara_geojson = boroondara.__geo_interface__

def base_figure():
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=nodes_df["lat"],
            lon=nodes_df["lon"],
            mode="markers",
            marker=dict(size=12, color="blue"),
            text=nodes_df["SCATS_Number"],
            hoverinfo="text+lat+lon",
            name="SCATS Nodes",
        )
    )

    fig.update_layout(  # Add Boroondara outline
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
app.layout = html.Div(
    [
        html.H1("Boroondara SCATS Path Finder"),
        html.Div(  # User input controls
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
                dcc.Input(id="sequence-length", type="number", value=12, min=1),
                html.Label("K Value"),
                dcc.Input(id="k-val", type="number", value=1, min=1),
                html.Button("Find Path", id="submit-button", n_clicks=0),
            ],
            style={"width": "30%", "display": "inline-block", "verticalAlign": "top"},
        ),
        html.Div(
            [
                dcc.Graph(id="map", figure=base_figure()),
            ],
            style={"width": "65%", "display": "inline-block"},
        ),
    ]
)


@app.callback(
    Output("map", "figure"),
    Input("submit-button", "n_clicks"),
    State("origin-input", "value"),
    State("destination-input", "value"),
    State("model-type", "value"),
    State("sequence-length", "value"),
    State("k-val", "value"),
)

def update_map(n_clicks, origin, destination, model_type, sequence_length, k_val):
    if n_clicks == 0:
        return base_figure()

    graph_builder = GraphVertexEdgeInit("./GUI/graph_init_data.xlsx")  # Build graph
    graph = graph_builder.extract_file_contents()

    current_time = datetime.now().replace(month=8)  # Fake current time as August same as the Demo

    path_finder = PathFinder(graph=graph)  # Create path finder object
    solution_nodes = path_finder.find_paths(  # Find paths
        initial_state=origin,
        goal_state=destination,
        initial_time=current_time,
        sequence_length=sequence_length,
        k_val=k_val,
        mode=model_type,
    )

    if not solution_nodes:
        return base_figure()

    best_path = solution_nodes[0]  # Take first solution path
    path_states = []
    node = best_path
    while node:
        path_states.append(node.state)
        node = node.parent
    path_states.reverse()

    coords = nodes_df.set_index("SCATS_Number").loc[path_states]  # Get coordinates for each state
    lats, lons = coords["lat"], coords["lon"]

    fig = base_figure()  # Build updated map
    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="lines+markers",
            line=dict(width=4, color="purple"),
            marker=dict(size=10, color="purple"),
            text=[str(s) for s in path_states],
            name="Optimal Path",
        )
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)