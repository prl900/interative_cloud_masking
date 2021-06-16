import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import plotly.express as px

import os
import xarray as xr
import numpy as np
from skimage import draw
from scipy import ndimage

#IMG_SIZE = 700
#MASK_SIZE = 550
IMG_SIZE = 800
MASK_SIZE = 450
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8

# the number of different classes for labels
NUM_LABEL_CLASSES = 2
DEFAULT_LABEL_CLASS = 1
class_label_colormap = ["#56B4E9", "#E69F00"]
color2class = {"#56B4E9": np.uint8(1), "#E69F00":np.uint8(2)}

class_labels = list(range(NUM_LABEL_CLASSES))

ds = xr.open_dataset("http://dapds00.nci.org.au/thredds/dodsC/ub8/au/blobs/sample_stack.nc")
MAX_ITIME = len(ds.time.values)

# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

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
    mask = {"#56B4E9": np.zeros(dims, dtype=np.bool), "#E69F00": np.zeros(dims, dtype=np.bool)}
    
    for shape in shapes:
        shape_mask = np.zeros(dims, dtype=np.uint8)
        cols, rows = path_to_indices(shape["path"]).T
        rr, cc = draw.polygon(rows, cols)
        rr = np.clip(rr, 0, 399)
        cc = np.clip(cc, 0, 399)
        shape_mask[rr, cc] = True
        shape_mask = ndimage.binary_fill_holes(shape_mask)
        mask[shape['line']['color']] += shape_mask

    return np.clip(mask["#56B4E9"]*color2class["#56B4E9"] + mask["#E69F00"]*color2class["#E69F00"], 0, 2)


def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = "Interactive image segmentation based on machine learning"


def ls_figure(
    itime=0,
    stroke_color=class_to_color(DEFAULT_LABEL_CLASS),
    stroke_width=int(round(2 ** (DEFAULT_STROKE_WIDTH))),
    shapes=[],
):
    img = ds.nbart_blue.isel(x=slice(400, 800), y=slice(0,400), time=itime).values
    fig = px.imshow(img, width=IMG_SIZE, height=IMG_SIZE)
    fig.update_layout(
        {
            "dragmode": "drawclosedpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig

def mask_figure(
    dims=(400,400),
    shapes=[],
):
    mask = path_to_mask(shapes, dims)
    fig = px.imshow(mask, width=MASK_SIZE, height=MASK_SIZE, range_color=[0.0, 3.0], color_continuous_scale=['black', '#56B4E9', '#E69F00', '#009E73'])
    fig.update_layout(coloraxis_showscale=False)

    return fig


# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader("Interactive Sentinel-2 Cloud Labelling: Select clouds and shadows in the series of images below"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                                    # Graph
                                    dcc.Graph(
                                        id="graph",
                                        figure=ls_figure(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawclosedpath",
                                                "eraseshape",
                                            ]
                                        },
                                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.ButtonGroup(id="nav-images", children=[dbc.Button("Prev", id={"type": "nav-images-button", "index": -1}), 
                                                                               dbc.Button("Next", id={"type": "nav-images-button", "index": 1})]),
                                ]
                            ),
                        ]
                    ),
                                #],
                            #)
                        ],
                    ),
                ]
            ),
        ],
    )
]

button_name = {0: "cloud", 1: "shadow"}

# sidebar
sidebar = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Tools"),
            dbc.CardBody(
                [
                    html.H6("Label class", className="card-title"),
                    # Label class chosen with buttons
                    html.Div(
                        id="label-class-buttons",
                        children=[
                            dbc.Button(
                                button_name[n],
                                id={"type": "label-class-button", "index": n},
                                style={"background-color": class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ],
                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Width of annotation paintbrush",
                                        html_for="stroke-width",
                                    ),
                                    # Slider for specifying stroke width
                                    dcc.Slider(
                                        id="stroke-width",
                                        min=0,
                                        max=6,
                                        step=0.1,
                                        value=DEFAULT_STROKE_WIDTH,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Label array",
                                        html_for="mask",
                                    ),
                                    dcc.Graph(
                                        id="mask",
                                        figure=mask_figure(),
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Button("Save", id="save-button", color="primary", className="mr-1", n_clicks=0),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
    ),
]

meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id="masks", data={"shapes": [[] for _ in range(MAX_ITIME)]}),
            dcc.Store(id="itime", data= 0),
        ],
    ),
]

app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    id="app-content",
                    children=[dbc.Col(segmentation, md=8), dbc.Col(sidebar, md=4)],
                ),
                dbc.Row(dbc.Col(meta)),
            ],
            fluid=True,
        ),
        html.Div(id='dummy')
    ]
)


@app.callback(
    Output("dummy", "children"),
    Input("save-button", "n_clicks"),
    State("masks", "data"),
)
def annotation_react(n_clicks, masks_data):
    if n_clicks == 0:
        raise PreventUpdate

    stack = np.empty((0,400,400), dtype=np.uint8)
    for i in range(MAX_ITIME):
        mask = path_to_mask(masks_data["shapes"][i], (400,400))
        stack = np.concatenate((stack, mask[None,:]), axis=0)

    #np.save("masks", stack)

    return


@app.callback(
    [
        Output("graph", "figure"),
        Output("mask", "figure"),
        Output("masks", "data"),
        Output("itime", "data"),
    ],
    [
        Input("graph", "relayoutData"),
        Input(
            {"type": "nav-images-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("stroke-width", "value"),
    ],
    [
        State("masks", "data"),
        State("itime", "data"),
    ],
)
def annotation_react(
    graph_relayoutData,
    image_nav_button_value,
    label_class_button_value,
    stroke_width_value,
    masks_data,
    itime_data,
):
    if callback_context.triggered[0]['prop_id'] == '{"index":-1,"type":"nav-images-button"}.n_clicks_timestamp':
        itime_data = max(0, itime_data-1)
    if callback_context.triggered[0]['prop_id'] == '{"index":1,"type":"nav-images-button"}.n_clicks_timestamp':
        itime_data = min(MAX_ITIME, itime_data+1)

    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"][itime_data] = graph_relayoutData["shapes"]
        else:
            return dash.no_update
    stroke_width = int(round(2**(stroke_width_value)))
    # find label class value by finding button with the most recent click
    if label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    fig = ls_figure(
        itime=itime_data,
        stroke_color=class_to_color(label_class_value),
        stroke_width=stroke_width,
        shapes=masks_data["shapes"][itime_data],
    )
    mask = mask_figure(
        (400,400),
        shapes=masks_data["shapes"][itime_data],
    )
    
    return (
        fig,
        mask,
        masks_data,
        itime_data,
    )


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True, use_reloader=False)
