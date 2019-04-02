# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:52:08 2019

@author: kennedy
"""

import pandas as pd
import dash
import os 
import time
import flask
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import plotly.graph_objs as go
from datetime import datetime


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Div([
        html.Div([
                html.H1('Digital Book Insight'),
                ], style={'text-align': 'left','width': '49%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Div([
                html.H4('Project by Miloskrissak'),
                html.Label('Dash is a web application framework that provides pure Python abstraction around HTML, CSS, and JavaScript.<Instead of writing HTML or using an HTML templating engine, you compose your layout using Python structures with the dash-html-components library.')
                ], style= {'width': '49%', 'display': 'inline-block','vertical-align': 'middle'})
                ], style={'background-color': 'white', 'box-shadow': 'black 0px 1px 0px 1px'}),
    html.Div([
            html.Div([
                    html.Label('x-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='x-items',
                            options = [
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Log', 'value': 'log'},
                                ],
                            value = "linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            html.Div([
                    html.Label('y-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='y-items',
                            options = [
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Log', 'value': 'log'},
                                ],
                            value = "linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            html.Div([
                    html.Label('X-Vals:'),                    
                    dcc.RadioItems(
                            #---
                            id='x-vals',
                            options = [
                                {'label': 'Views', 'value': 'views'},
                                {'label': 'Duration', 'value': 'Duration'},
                                ],
                            value = "views",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            html.Div([
                    html.Label('Sort Tags'),                    
                    dcc.RadioItems(
                            #---
                            id='Sort-Tags',
                            options = [
                                {'label': 'A-z', 'value': 'A-z'},
                                {'label': 'Most Tags', 'value': 'Most Tags'},
                                ],
                            value = "A-z",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'})
            ], style={'background-color': 'rgb(204, 230, 244)', 'padding': '1rem 0px', 'margin-top': '2px'})
    ],style = {'margin': 'auto', 'width': '100%', 'box-sizing': 'border-box'})



if __name__ == '__main__':
  app.run_server(debug = True)
  

  