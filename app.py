#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:23:37 2019

@author: kenneth
"""

import pandas as pd
import dash
import os 
import nltk
import numpy as np
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from os.path import join
from collections import Counter
nltk.download('inaugural')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#-config tools
config={
        "displaylogo": False,
        'modeBarButtonsToRemove': ['pan2d','lasso2d', 'hoverClosestCartesian',
                                   'hoverCompareCartesian', 'toggleSpikelines',
                                   ]
    }
#%% data

#get file path
path = '/app'
direc = join(path, 'DATASET/')
data = pd.read_csv(direc + 'collatedsources_v1.csv', sep = ';')
data.set_index(['ID'], inplace = True)
columns = [x for x in data.columns]

#-------------------------------

book_path = join(path, 'DATASET/Collated books v1/')
dirlis = sorted(os.listdir(book_path))[1:]


#%%
def tokenize(token_len):
    book_path = join(path, 'DATASET/')
    dirlis = sorted(os.listdir(book_path+'Collated books v1/'))[1:]
    stopwords = set(nltk.corpus.stopwords.words('french'))
    sentence = []
    for ii in dirlis:
        with open(book_path+'Collated books v1/'+ii, 'r+') as file:
            file_dt = file.read()
            #tokenize and stem
            tokenizer = RegexpTokenizer(r'\w+')
            up_text = tokenizer.tokenize(file_dt)
            file.close()
            #--unprocessed tokens
            if not os.path.exists(join(book_path+'token/', ii.strip('.txt')+str('_new.txt'))):
                with open(join(book_path+'token/', ii.strip('.txt')+str('_new.txt')), 'w') as wr:
                    wr.writelines('\n'.join(up_text))
            else:
                pass
            #--processed tokens
            new_token = []
            new_words = [ii for ii in up_text if len(ii) >= int(token_len)]
            for each_word in new_token:
                each_word = each_word.lower()
                if each_word not in stopwords:
                    new_words.append(each_word)
            final = ' '.join(new_words)
            sentence.append(str(final))
    
#    file = pd.DataFrame({'text': sentence, 'category': data.book_category_name.values})
#    file.to_csv(book_path+'ptoken/'+'ptoken.csv')
    return sentence

#--preprocess for brief display
def preprocess():
    tokenizer = RegexpTokenizer(r'\w+')
    book_path = join(path, 'DATASET/')
    dirlis = sorted(os.listdir(book_path + 'Collated books v1/'))[1:]
    for ii in dirlis:
            with open(book_path + 'Collated books v1/' + ii, 'r') as file:
                text = file.read().strip()[0:500]
                text = tokenizer.tokenize(text)
                text = ' '.join(text)
                file.close()
                if not os.path.exists(join(book_path+'filtered_book/', ii)):
                    with open(join(book_path+'filtered_book/', ii), 'w+') as wr:
                        wr.writelines(text)
                else:
                    pass


sentences = tokenize(5)
preprocess()

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(sentences)
tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)


#%% app

app.layout = html.Div([
    html.Div([
        #--header section
        html.Div([
                html.H1('Digital Book Insight'),
                ], style={'text-align': 'left','width': '49%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Div([
                html.H4('Project by E. kenneth'),
                html.Label('NLP with python 3: Topic visualization in intereative chart. Cluster analysis of word corpus using k-means algorithm. Hover over the data points to see '+
                           'meta data info on respective books.')
                ], style= {'width': '49%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '12px'})
                ], style={'background-color': 'white', 'box-shadow': 'black 0px 1px 0px 0px'}),
    #--scaling section
    html.Div([
            html.Div([
                    html.Label('Cluster size: Default is optimum'),                    
                    dcc.RadioItems(
                            #---
                            id='cluster',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(2, 6, 1)]],
                            value = "3",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            html.Div([
                    html.Label('y-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='y-items',
                            options = [{'label': i, 'value': i} for i in ['Linear', 'Log']],
                            value = "Linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            #--- Token length
            html.Div([
                    html.Label('Token length:'),                    
                    dcc.RadioItems(
                            #---
                            id='tokens',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(5, 11, 1)]],
                            value = "5",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'}),
            #--- Sort Tags
            html.Div([
                    html.Label('Sort Tags'),                    
                    dcc.RadioItems(
                            #---
                            id='Sort-Tags',
                            options = [{'label': i, 'value': i} for i in ['A-z', 'Most Tags']],
                            value = "A-z",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '25%'})
            ], style={'background-color': 'rgb(204, 230, 244)', 'padding': '1rem 0px', 'margin-top': '2px','box-shadow': 'black 0px 0px 1px 0px','vertical-align': 'middle'}),
    #-- Graphs
    html.Div([
            html.Div([
            #--scatterplot
            #visibility: visible; left: 0%; width: 100%
#            html.Div([

                    dcc.Dropdown(
                        #---
                        id='dd',
                        options =[{'label': i, 'value': i} for i in list(data.book_category_name.unique())],
                        value = [],
                        placeholder = 'Select a category',
                        multi = True,
                        ),
                   dcc.Graph(id = 'scatter_plot',
#                              style={'width': '690px', 'height': '395px'},
                      config = config,
                      hoverData={'points': [{'customdata': ["06_07", "Paris", "Broussais, F.-J.-V.", "Histoire des phlegmasies ou inflammations chroniques (2 vols.)", 1808]}]}
                      ),
#                    ], style = {'display': 'inline-block', 'width': '65%','background-color': 'white'}),
        
            ],style = {'display': 'inline-block', 'background-color': 'white', 'width': '65%', 'padding': '0 20','vertical-align': 'middle'}),
    #--horizontal dynamic barplot
    html.Div([
            dcc.Graph(id = 'bar_plot',
                      config = config,
                      )
            ],style = {'display': 'inline-block', 'background-color': 'white', 'width': '35%','vertical-align': 'middle'}),
    html.Div([
            dcc.RangeSlider(
                    id='year-slider',
                    min=data.year_edited.min(),
                    max=data.year_edited.max(),
                    value = [data.year_edited.min(), data.year_edited.max()],
                    marks={str(year): str(year) for year in range(data.year_edited.min(), data.year_edited.max(), 5)}
                ),
            ], style = {'background-color': 'white', 'display': 'inline-block', 'width': '65%', 'padding': '0px 20px 20px 20px','vertical-align': 'middle'}),
            ], style = {'background-color': 'white','margin': 'auto', 'width': '100%', 'display': 'inline-block'}),
    
    #-- Footer section
    html.Div([
        #--footer section
        html.Div([
                html.Div([
                        html.H2(id = 'topic')], style = {'color':' rgb(35, 87, 137)'}),
                html.Div([
                        html.Label(id = 'date')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block'}),
                html.Div([
                        html.Label(id = 'author')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block', 'padding': '0px 0px 10px 35px'}),
                html.Div([
                        html.Label(id = 'cat')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block', 'padding': '0px 0px 10px 35px'}),
                html.Label(id = 'label'),
                ], style= {'width': '74%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '15px'}),
        html.Div([
                html.H2('Topics'),
                html.Label('Dash is a web application framework that provides pure Python')
                ], style={'text-align': 'center','width': '25%', 'display': 'inline-block','vertical-align': 'middle'}),
                ], style={'background-color': 'rgb(204, 230, 244)', 'margin': 'auto', 'width': '100%', 'max-width': '1200px', 'box-sizing': 'border-box', 'height': '30vh'}),
    #---
    #main div ends here
    ],style = {'background-color': 'rgb(204, 230, 244)','margin': 'auto', 'width': '100%', 'display': 'block'})

#--
@app.callback(
        Output('scatter_plot', 'figure'),
        [Input('year-slider', 'value'),
         Input('dd', 'value'),
         Input('y-items', 'value'),
         Input('cluster', 'value')])
def update_figure(make_selection, drop, yaxis, clust):
    data_places = data[(data.year_edited >= make_selection[0]) & (data.year_edited <= make_selection[1])]
    if drop != []:
        traces = []
        for val in drop:
            traces.append(go.Scatter(
                    x = data_places.loc[data_places['book_category_name'] == str(val), 'year_edited'],
                    y = similarity_df.iloc[:, 0].values,
                    text = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places.loc[data_places['book_category_name'] == str(val), 'book_code'],\
                             data_places.loc[data_places['book_category_name'] == str(val), 'place'],\
                            data_places.loc[data_places['book_category_name'] == str(val), 'author'], \
                            data_places.loc[data_places['book_category_name'] == str(val), 'book_title'] ,\
                            data_places.loc[data_places['book_category_name'] == str(val), 'year_edited'])],
                    customdata = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places.loc[data_places['book_category_name'] == str(val), 'book_code'],\
                             data_places.loc[data_places['book_category_name'] == str(val), 'place'],\
                            data_places.loc[data_places['book_category_name'] == str(val), 'author'], \
                            data_places.loc[data_places['book_category_name'] == str(val), 'book_title'] ,\
                            data_places.loc[data_places['book_category_name'] == str(val), 'year_edited'])],
                    mode = 'markers',
                    opacity = 0.6,
                    marker = {'size': 15, 
#                              'color': 'rgba(50, 171, 96, 0.6)',
                              'line': {'width': 0.5, 'color': 'white'}},
                    name = val,
                    ))
        
        return {'data': traces,
                'layout': go.Layout(
#                        height = 600,
                        xaxis={'title': 'year'},
                        yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'Similarity score'},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 0, 'y': 1},
                        hovermode='closest')
                        }
    else:
        km = KMeans(n_clusters = int(clust), init = 'k-means++')
        km.fit_transform(similarity_df)
        cluster_labels = km.labels_
        traces = go.Scatter(
                x = data_places['year_edited'],
                y = similarity_df.iloc[:, 0].values,
                text = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places['book_code'], data_places['place'],\
                        data_places['author'], data_places['book_title'] , data_places['year_edited'])],
                customdata = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places['book_code'], data_places['place'],\
                        data_places['author'], data_places['book_title'] , data_places['year_edited'])],
                mode = 'markers',
                opacity = 0.7,
                marker = {'size': 15, 
#                          'opacity': 0.9,
                          'color': cluster_labels,
                          'line': {'width': .5, 'color': 'white'}},
                )
        
        return {'data': [traces],
                'layout': go.Layout(
                        height = 600,
                        xaxis={'title': 'year'},
                        yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'Similarity score'},
                        
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 0, 'y': 1},
                        hovermode='closest')
                        }

@app.callback(
        Output('topic', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookheader(hoverData):
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/Collated books v1/')
    dirlis = sorted(os.listdir(book_path))[1:]
    for ii in dirlis:
        if ii.strip('.txt') == book_number:
            subject = data[data.book_code == ii.strip('.txt')]['book_title'].values[0]
    return subject

@app.callback(
        Output('date', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookyear(hoverData):
    
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/Collated books v1/')
    dirlis = sorted(os.listdir(book_path))[1:]
    for ii in dirlis:
        if ii.strip('.txt') == book_number:
            date = data[data.book_code == ii.strip('.txt')]['year_edited'].values[0]
    return str('YEAR EDITED: ')+ str(date)

@app.callback(
        Output('author', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookauthor(hoverData):
    
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/Collated books v1/')
    dirlis = sorted(os.listdir(book_path))[1:]
    for ii in dirlis:
        if ii.strip('.txt') == book_number:
            author = data[data.book_code == ii.strip('.txt')]['author'].values[0]
    return str('Author: ') + str(author)

@app.callback(
        Output('cat', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_cat(hoverData):
    
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/Collated books v1/')
    dirlis = sorted(os.listdir(book_path))[1:]
    for ii in dirlis:
        if ii.strip('.txt') == book_number:
            cat = data[data.book_code == ii.strip('.txt')]['book_category_name'].values[0]
    return str('Category: ') + str(cat)
    
@app.callback(
        Output('label', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_label(hoverData):
    #--
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/filtered_book/')
    dirlis = sorted(os.listdir(book_path))
    for ii in dirlis:
        if ii.strip('.txt') == book_number:
            with open(join(book_path, ii), 'r+') as file:
                text = file.read()
                file.close()
    return text

@app.callback(
        Output('bar_plot', 'figure'),
        [Input('scatter_plot', 'hoverData'),
         Input('Sort-Tags', 'value'),
         Input('tokens', 'value')]
        )
def bar_plot(hoverData, sort, token):
    #--locate book and extract data from drive
    book_number = hoverData['points'][0]['customdata'][0]
    book_path = join(path, 'DATASET/token/')
    dirlis = sorted(os.listdir(book_path))
    stopwords = set(nltk.corpus.stopwords.words('french'))
    freq_word  = Counter()
    result=[]
    for ii in dirlis:
        if ii.strip('_new.txt') == book_number:
            with open(join(book_path, ii), 'r+') as wr:
                file = [wr.strip() for wr in wr.readlines()]
#                wr.close()
                for tok in file:
                    if len(tok) > int(token):
                        result.append(tok)
                    else:
                        pass
            for word in result:
                # update count off all words in the line that are in stopwords
                if word not in stopwords:
                    freq_word.update([word])
            #--trace
            trac_x, trac_y = [], []
            for w, y in freq_word.most_common(15):
                trac_x.append(y)
                trac_y.append(w)
            trace = go.Bar(
                    x = trac_x,
                    y = trac_y,
                    marker = dict(
                        color='rgba(50, 171, 96, 0.6)',
                        line=dict(
                            color='rgba(50, 171, 96, 1.0)',
                            width=2),
                    ),
                    orientation = 'h',
                    )
                   
            return {'data': [trace],
                    'layout': go.Layout(
                            autosize  =False,
                            width = 500,
                            height = 600,
                            margin=go.layout.Margin(
                                    l=100,
                                    r=50,
                                    b=100,
                                    t=0,
                                    pad=4
                                    ),
                            yaxis={'autorange': 'reversed' if sort == 'A-z' else True},
                            )
                            
                    }


if __name__ == '__main__':
  app.run_server(debug = True)






