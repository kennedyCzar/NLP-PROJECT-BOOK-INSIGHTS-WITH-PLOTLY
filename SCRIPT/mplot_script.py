# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:52:08 2019

@author: kennedy
"""

import pandas as pd
import dash
import os 
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import plotly.graph_objs as go
from datetime import datetime
from os.path import join

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#get file path
path = '/home/kenneth/Documents/GIT_PROJECTS/NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY'
direc = join(path, 'DATASET/')
data = pd.read_csv(direc + 'collatedsources_v1.csv', sep = ';')
data.set_index(['ID'], inplace = True)

app.layout = html.Div([
    html.Div([
        #--header section
        html.Div([
                html.H1('Digital Book Insight'),
                ], style={'text-align': 'left','width': '49%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Div([
                html.H4('Project by Miloskrissak'),
                html.Label('Dash is a web application framework that provides pure Python abstraction around HTML, CSS, and JavaScript.<Instead of writing HTML or using an HTML templating engine, you compose your layout using Python structures with the dash-html-components library.')
                ], style= {'width': '49%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '12px'})
                ], style={'background-color': 'white', 'box-shadow': 'black 0px 1px 0px 0px'}),
    #--scaling section
    html.Div([
            #--- x-axis
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
            #--- y-axis
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
            #--- X-Vals
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
            #--- Sort Tags
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
            ], style={'background-color': 'rgb(204, 230, 244)', 'padding': '1rem 0px', 'margin-top': '2px','box-shadow': 'black 0px 0px 1px 0px'}),
    #-- Graphs
    html.Div([
            #--scatterplot
            #visibility: visible; left: 0%; width: 100%
            html.Div([
                    dcc.Graph(id = 'scatter_plot'),
                    ], style = {'display': 'inline-block', 'width': '65%'}),
            #--horizontal dynamic barplot
            html.Div([
                    dcc.Graph(id = 'bar_plot')
                    ], style = {'display': 'inline-block', 'width': '35%'}),
            ]),
    html.Div([
            dcc.Slider(
                    id='year-slider',
                    min=data.year_edited.min(),
                    max=data.year_edited.max(),
                    value=data.year_edited.min(),
                    marks={str(year): str(year) for year in range(data.year_edited.min(), data.year_edited.max(), 5)}
                )
            ], style = {'background-color': 'rgb(204, 230, 244)', 'visibility': 'visible', 'left': '0%', 'width': '49%', 'padding': '0px 20px 20px 20px'}),
    #-- Footer section
    html.Div([
        #--footer section
        
        html.Div([
                html.H4('Project by Miloskrissak'),
                html.Label('Dash is a web application framework that provides pure Python abstraction around HTML, CSS, and JavaScript.<Instead of writing HTML or using an HTML templating engine, you compose your layout using Python structures with the dash-html-components library.')
                ], style= {'width': '74%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '12px'}),
        html.Div([
                html.H6('Digital Book Insight'),
                html.Label('Dash is a web application framework that provides pure Python')
                ], style={'text-align': 'center','width': '25%', 'display': 'inline-block','vertical-align': 'middle'}),
                ], style={'background-color': 'rgb(204, 230, 244)', 'margin': 'auto', 'width': '100%', 'max-width': '1200px', 'box-sizing': 'border-box', 'height': '30vh'}),
    #---
    #main div ends here
    ],style = {'background-color': 'rgb(204, 230, 244)','margin': 'auto', 'width': '100%', 'display': 'block'})

@app.callback(
        Output('scatter_plot', 'figure'),
        [Input('year-slider', 'value')]
        )
def update_figure(make_selection):
    year_filter = data[data.year_edited == make_selection]
    traces = []
    for ii in year_filter.place.unique():
        data_places = year_filter[year_filter['place'] == ii]
        traces.append(go.Scatter(
                x = data_places.index,
                y = data_places['book_number'],
                text = data_places['place'],
                mode = 'markers',
                opacity = 0.5,
                marker = {'size': 10, 'line': {'width': 0.5, 'color': 'white'}},
                name = ii,
                ))
    return {'data': traces,
            'layout': go.Layout(
                    xaxis={'type': 'linear', 'title': 'Book ID'},
                    yaxis={'title': 'Book number'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest')
                    }
        
if __name__ == '__main__':
  app.run_server(debug = True)
  

  