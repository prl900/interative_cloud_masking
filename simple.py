import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Dash, ServersideOutput
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import plotly.express as px

from random import randint
import os
import xarray as xr
import numpy as np
from skimage import draw
from skimage.metrics import structural_similarity as ssim

from scipy import ndimage

IMG_SIZE = 800

def load_dataset(i, j):
    print(i,j)
    dsa = xr.open_dataset(f"/data/act_dea_merge/s2a_{i:02d}_{j:02d}_2020.nc")
    dsa = dsa.rename_vars(name_dict={'nbart_nir_1':'nbart_nir','nbart_swir_2':'nbart_swir_1','nbart_swir_3':'nbart_swir_2'})
    dsb = xr.open_dataset(f"/data/act_dea_merge/s2b_{i:02d}_{j:02d}_2020.nc")
    dsb = dsb.rename_vars(name_dict={'nbart_nir_1':'nbart_nir','nbart_swir_2':'nbart_swir_1','nbart_swir_3':'nbart_swir_2'})
    ds7 = xr.open_dataset(f"/data/act_dea_merge/ls7_{i:02d}_{j:02d}_2020.nc")
    ds8 = xr.open_dataset(f"/data/act_dea_merge/ls8_{i:02d}_{j:02d}_2020.nc")
    ds = xr.concat([dsa, dsb, ds7, ds8], dim='time').sortby('time')
    ds = ds.dropna('time', how='all')

    return ds


def filter_dataset(ds, percentile=0.3, threshold=0.5):
    print(ds)
    p30 = ds.nbart_blue.quantile(percentile, dim='time').values.astype(np.float32)
    ssi = np.empty(ds.nbart_blue.shape)
    print(ds.nbart_blue.shape)
    for i in range(len(ds.time)):
        im = np.copy(ds.nbart_blue.isel(time=i).values)
        #if np.count_nonzero(np.isnan(im)) < im.size*0.5:
        im[np.isnan(im)] = p30[np.isnan(im)]
        _, ssi_full = ssim(p30, im, full=True)
        ssi_full[np.isnan(im)] = np.nan
        ssi[i,:] = (ssi_full + 1)/2
        #else:
        #    ssim_idxs.append(0.0)

    ds['ssi'] = (['time', 'y', 'x'], ssi)

    return ds

def rgb_fig(ds, i):
    im = np.moveaxis(ds[["nbart_red","nbart_green","nbart_blue"]].isel(time=i).to_array().values, 0, -1)
    im = np.clip(im, 0, 3000)
    im = ((im / 3000) * 255).astype(np.uint8)
    print(im.shape)
    #return px.imshow(np.moveaxis(ds[["nbart_red","nbart_green","nbart_blue"]].isel(time=i).to_array().values, 0, -1), width=IMG_SIZE, height=IMG_SIZE, zmin=0, zmax=3000, color_continuous_scale='RdBu_r')    
    print("A")
    return px.imshow(im, width=IMG_SIZE, height=IMG_SIZE)#, zmin=0, zmax=3000, color_continuous_scale='RdBu_r')    
    
def ssi_fig(ds, i):
    print("B")
    return px.imshow(ds.ssi.isel(time=i), width=IMG_SIZE, height=IMG_SIZE, zmin=0, zmax=1, color_continuous_scale='RdBu_r')


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = "Interactive image segmentation based on machine learning"


# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader("Interactive Sentinel-2 Cloud Labelling: Select clouds and shadows in the series of images below"),
            dbc.CardBody([
                # Wrap dcc.Loading in a div to force transparency when loading
                dbc.Row([
                    html.Div(dcc.Loading(dcc.Graph(
                        id="graph1"
                    ))),
                    html.Div(dcc.Loading(dcc.Graph(
                        id="graph2"
                    ))),
                ]),
                html.Hr(),
                dbc.Button("Random", id="load-ds", color="info", className="mr-1", n_clicks=0),
                dbc.Form([
                        dbc.FormGroup([
                                dbc.ButtonGroup(id="nav-images", children=[dbc.Button("Prev", id={"type": "nav-images-button", "index": -1}), 
                                dbc.Button("Next", id={"type": "nav-images-button", "index": 1})]),
                        ]),
                ]),
            ]),
        ]
    ),
]

app.layout = html.Div(
    [
        dcc.Store(id="store"),
        dcc.Store(id="store2"),
        dbc.Container(
            [
                dbc.Row(
                    id="app-content",
                    children=[dbc.Col(segmentation, md=12)],
                ),
            ],
            fluid=True,
        ),
        html.Div(id='dummy')
    ]
)

@app.callback(ServersideOutput("store", "data"), ServersideOutput("store2", "data"), Input("load-ds", "n_clicks"))
def query(_):
    ds = load_dataset(randint(0,10), randint(0,20))
    ds2 = filter_dataset(ds, percentile=0.3, threshold=0.5)
    return ds, ds2


@app.callback(Output("graph1", "figure"), Output("graph2", "figure"), Input("store2", "data"))
def right(ds):
    return rgb_fig(ds, 1), ssi_fig(ds, 1)

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 8888)), debug=True, use_reloader=False)
