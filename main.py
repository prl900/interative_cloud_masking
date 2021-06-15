import numpy as np
import xarray as xr

import plotly.express as px
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from skimage import data, draw
from scipy import ndimage

class_labels = [1, 2]
class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

def class_to_color(n):
    return class_label_colormap[n]

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


def path_to_mask(shapes, dims):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    mask = np.zeros(dims, dtype=np.bool)
    for shape in shapes:#last_shape = relayout_data["shapes"]#[-1]
        cols, rows = path_to_indices(shape["path"]).T
        rr, cc = draw.polygon(rows, cols)
        mask[rr, cc] = True

    mask = ndimage.binary_fill_holes(mask)
    return mask


ds = xr.open_dataset("sample_stack.nc")
img = ds.nbart_blue.isel(time=1).values[:400, 400:800]
fig = px.imshow(img, binary_string=True, width=800, height=800)
fig.update_layout(dragmode="drawclosedpath")

fig_mask = px.imshow(np.zeros((400, 400), dtype=np.uint8), binary_string=True, width=800, height=800)

fig_hist = px.histogram(img.ravel())

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H3("Draw a path to delimit the areas covered by clouds or shadows"),
        html.Div(
            [dcc.Graph(id="graph-image", figure=fig),],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div(
            [dcc.Graph(id="graph-mask", figure=fig_mask),],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div([html.H6("Label class", className="card-title"),
                    # Label class chosen with buttons
                html.Div(
                    id="label-class-buttons",
                    children=[
                        dbc.Button(
                            "%2d" % (n,),
                            id={"type": "label-class-button", "index": n},
                            style={"background-color": class_to_color(c)},
                        )
                        for n, c in enumerate(class_labels)
                    ],
                ),
                html.Hr(), dcc.Graph(id="graph-histogram", figure=fig_hist)],
            style={"width": "20%", "display": "inline-block", "padding": "0 0"},
        ),
    ]
)


@app.callback(
    [Output("graph-histogram", "figure"), Output("graph-mask", "figure")],
    Input("graph-image", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]#[-1]
        mask = path_to_mask(shapes, img.shape)
        return px.histogram(img[mask]), px.imshow(mask, width=800, height=800)
    else:
        return dash.no_update, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
