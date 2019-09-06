# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:52:08 2019

@author: kennedy
"""

#################################################################################
#MIT License
#
#Copyright (c) 2019 KennedyWolfie
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#################################################################################

import pandas as pd
import dash
import os 
import nltk
import random
import numpy as np
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from os.path import join
from flask import Flask
from collections import Counter
from nltk.tokenize import RegexpTokenizer

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server = server)

#-config tools
config={
        "displaylogo": False,
        'modeBarButtonsToRemove': ['pan2d','lasso2d', 'hoverClosestCartesian',
                                   'hoverCompareCartesian', 'toggleSpikelines',
                                   ]
    }


#%% data

#get file path
path = '/home/kenneth/Documents/GIT_PROJECTS/NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY'
direc = join(path, 'DATASET/')
data = pd.read_csv(direc + 'collatedsources_v1.csv', sep = ';')
data.set_index(['ID'], inplace = True)
columns = [x for x in data.columns]


#load stopwords from drive
with open(join(path, 'stopwords'), 'r+') as st:
    stopwords = [x for x in st.read().split()]
    
#-------------------------------

#book_path = join(path, 'DATASET/Collated books v1/')
#dirlis = sorted(os.listdir(book_path))[1:]


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
            for each_word in new_words:
                each_word = each_word.lower()
                if each_word not in stopwords:
                    new_token.append(each_word)
            final = ' '.join(new_words)
            sentence.append(str(final))
    #--save files to directory
    if not os.path.exists(join(book_path, 'ptoken/ptoken.csv.gz')):
        file = pd.DataFrame({'text': sentence, 'year_edited': data.year_edited, 
                             'book_category_name': data.book_category_name})
        file.to_csv(book_path+'ptoken/'+'ptoken.csv')


#--preprocessing for brief display
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


#tokenize(5)
#preprocess()
sentences = list(pd.read_csv(join(path, 'DATASET/ptoken/ptoken.csv.gz'))['text'])
#

#%%

#tokenize(5) 
#preprocess() 
#sentences = list(pd.read_csv(join(path, 'DATASET/ptoken/ptoken.csv')).sort_values(by=['book_category_code', 'year_edited'])['text']) 
#vb = pd.read_csv(join(path, 'DATASET/ptoken/ptoken.csv')).sort_values(by=['year_edited']) 
sort_dataset = data.sort_values(by=['book_category_code', 'year_edited']) 
## 
## 
#datset = [] 
#for ii in data.book_category_name.unique(): 
#    datset.append(sort_dataset[sort_dataset.book_category_name == ii]) 
#     
#cat_1, cat_2, cat_3, cat_4, cat_5, cat_6 = [], [], [], [], [], [] 
# 
#for ij in range(len(datset)): 
#    bk_code = datset[ij].book_code.values 
#    for ii in bk_code: 
#        with open(direc + 'Collated books v1/' + ii + '.txt', 'r+') as file: 
#            file_dt = file.read() 
#            #tokenize and stem 
#            tokenizer = RegexpTokenizer(r'\w+') 
#            up_text = tokenizer.tokenize(file_dt) 
#            #--processed tokens 
#            new_token = [] 
#            new_words = [ii for ii in up_text if len(ii) >= int(5)] 
#            for each_word in new_words: 
#                each_word = each_word.lower() 
#                if each_word not in stopwords: 
#                    new_token.append(each_word) 
#            final = ' '.join(new_token) 
#            if ij == 0: 
#                cat_1.append(final) 
#            elif ij == 1: 
#                cat_2.append(final) 
#            elif ij == 2: 
#                cat_3.append(final) 
#            elif ij == 3: 
#                cat_4.append(final) 
#            elif ij == 4: 
#                cat_5.append(final) 
#            elif ij == 5: 
#                cat_6.append(final) 
# 
#if not os.path.exists(join(direc, 'category/cat_1.csv')): 
#    cat_1 = pd.DataFrame({'text': cat_1}) 
#    cat_1.to_csv(direc+'category/'+'cat_1.csv') 
# 
#if not os.path.exists(join(direc, 'category/cat_2.csv')): 
#    cat_2 = pd.DataFrame({'text': cat_2}) 
#    cat_2.to_csv(direc+'category/'+'cat_2.csv') 
# 
#if not os.path.exists(join(direc, 'category/cat_3.csv')): 
#    cat_3 = pd.DataFrame({'text': cat_3}) 
#    cat_3.to_csv(book_path+'category/'+'cat_3.csv') 
# 
#if not os.path.exists(join(direc, 'category/cat_4.csv')): 
#    cat_4 = pd.DataFrame({'text': cat_4}) 
#    cat_4.to_csv(direc+'category/'+'cat_4.csv') 
#     
#if not os.path.exists(join(direc, 'category/cat_5.csv')): 
#    cat_5 = pd.DataFrame({'text': cat_5}) 
#    cat_5.to_csv(direc+'category/'+'cat_5.csv') 
#     
#if not os.path.exists(join(direc, 'category/cat_6.csv')): 
#    cat_6 = pd.DataFrame({'text': cat_6}) 
#    cat_6.to_csv(direc+'category/'+'cat_6.csv') 
# 
# 
##%% 
#from sklearn.feature_extraction.text import TfidfVectorizer 
#from sklearn.metrics.pairwise import cosine_similarity 
#from sklearn.cluster import KMeans 
#from sklearn.decomposition import PCA 
# 
#def similarity_score(df): 
#    tv = TfidfVectorizer(min_df=1, use_idf=True) 
#    tv_matrix = tv.fit_transform(df) 
#    tv_matrix = tv_matrix.toarray() 
##    vocab = tv.get_feature_names() 
#    similarity_matrix = cosine_similarity(tv_matrix) 
#    similarity_df = pd.DataFrame(similarity_matrix) 
#    return similarity_df 
# 
#siml_1 = similarity_score(cat_1) 
#siml_2 = similarity_score(cat_2) 
#siml_3 = similarity_score(cat_3) 
#siml_4 = similarity_score(cat_4) 
#siml_5 = similarity_score(cat_5) 
#siml_6 = similarity_score(cat_6) 
# 
#similary_sc = pd.concat([siml_1[0], siml_2[0], siml_3[0], siml_4[0], siml_5[0], siml_6[0]], axis = 0) 
#similary_sc.to_csv(direc + 'similarity_score') 
#%% 
similary_sc = pd.read_csv(join(direc, 'similarity_score'), names = ['score']) 
 
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA

tv = TfidfVectorizer(min_df=5, use_idf=True)
tv_matrix = tv.fit_transform(sentences)
tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)

#%%


##topic modeling
#lda = LatentDirichletAllocation(n_components=2, max_iter=2, random_state=0)
#dt_matrix = lda.fit_transform(tv_matrix)
#features = pd.DataFrame(dt_matrix, columns=['T1', 'T2'])
#tt_matrix = lda.components_
#
#for topic_weights in tt_matrix:
#    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
#    topic = sorted(topic, key=lambda x: -x[1])
#    topic = [item for item in topic if item[1] > 0.9]
#    print(topic)
#    print()


#%%


#book_path = join(path, 'DATASET/token/')
#dirlis = sorted(os.listdir(book_path))
#with open(book_path + dirlis[1], 'r') as f:
#    file = [wr.strip() for wr in f.readlines()]
#    file = [x for x in file if x not in stopwords and len(x)>3]
#    
#words = [x.lower() for x in file]
#processed_docs = [file]
#dictionary = gensim.corpora.Dictionary(processed_docs)
#bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#
#ldamodel = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=1, id2word = dictionary, passes=1)
#
#for _, topic in ldamodel.show_topics(formatted=False, num_words= 10):
#    result = [w[0] for w in topic if w[0] not in stopwords]
#    result = ','.join(result)
    
    
    
#%% app

app.layout = html.Div([
    html.Div([
        #--header section
        html.Div([
                html.H1('Medical Book clustering'),
                ], style={'text-align': 'left','width': '49%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Div([
                html.H4('Project by E. kenneth'),
                html.Label('NLP with Plotly Dash: An unsupervised approach to clustering medical acheological books for insight discovery.')
                ], style= {'width': '49%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '12px'})
                ], style={'background-color': 'white', 'box-shadow': 'black 0px 1px 0px 0px'}),
    #--scaling section
    html.Div([
            html.Div([
                    html.Label('Cluster size: Default is optimum'),                    
                    dcc.RadioItems(
                            #---
                            id='cluster',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(2, 7, 1)]],
                            value = "3",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'}),
            html.Div([
                    html.Label('Number of Topics:'),                    
                    dcc.RadioItems(
                            #---
                            id='topic-number',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(5, 11, 1)]],
                            value = "5",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'}),
            html.Div([
                    html.Label('y-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='y-items',
                            options = [{'label': i, 'value': i} for i in ['Linear', 'Log']],
                            value = "Linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'}),
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
                    ], style = {'display': 'inline-block', 'width': '20%'}),
            #--- Sort Tags
            html.Div([
                    html.Label('Sort Tags'),                    
                    dcc.RadioItems(
                            #---
                            id='Sort-Tags',
                            options = [{'label': i, 'value': i} for i in ['A-z', 'Most Tags']],
                            value = "Most Tags",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'})
            ], style={'background-color': 'rgb(204, 230, 244)', 'padding': '1rem 0px', 'margin-top': '2px','box-shadow': 'black 0px 0px 1px 0px','vertical-align': 'middle'}),
    #-- Graphs
    html.Div([
            html.Div([
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
                    updatemode='drag',
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
#                html.Label(id = 'topic-tags'),
                html.Label(id = 'topic-tags', style={'text-align': 'center', 'margin': 'auto', 'vertical-align': 'middle'})
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
         Input('cluster', 'value'),
         ])
def update_figure(make_selection, drop, yaxis, clust):
#    data_places = data[(data.year_edited >= make_selection[0]) & (data.year_edited <= make_selection[1])]
    data_places = sort_dataset[(sort_dataset.year_edited >= make_selection[0]) & (sort_dataset.year_edited <= make_selection[1])] 
    if drop != []:
        traces = []
        for val in drop:
            traces.append(go.Scattergl(
                    x = data_places.loc[data_places['book_category_name'] == str(val), 'year_edited'],
                    y = similarity_df[0].values,
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
                        legend={'x': 1, 'y': 1},
                        hovermode='closest')
                        }
    else:
        pca = PCA(n_components = int(clust)).fit(similarity_df)
        km = KMeans(n_clusters = int(clust), init = pca.components_, n_init = 1)
        km.fit_transform(similarity_df)
        cluster_labels = km.labels_
        traces = go.Scattergl(
                x = data_places['year_edited'],
                y = similary_sc['score'].values[1:],
                text = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places['book_code'], data_places['place'],\
                        data_places['author'], data_places['book_title'] , data_places['year_edited'])],
                customdata = [(x, y, z, w, q) for (x, y, z, w, q) in zip(data_places['book_code'], data_places['place'],\
                        data_places['author'], data_places['book_title'] , data_places['year_edited'])],
                mode = 'markers',
                opacity = 0.7,
                marker = {'size': 15, 
#                          'opacity': 0.9,
                          'color': cluster_labels,
                          'colorscale':'Viridis',
                          'line': {'width': .5, 'color': 'white'}},
                )
        
        return {'data': [traces],
                'layout': go.Layout(
                        height = 600,
                        xaxis={'title': 'year'},
                        yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'Similarity score'},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 1, 'y': 1},
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
        Output('topic-tags', 'children'),
        [Input('scatter_plot', 'hoverData'),
         Input('tokens', 'value'),
         Input('topic-number', 'value')]
        )
def topic_tags(hoverData, token, topic):
    #--
#    import random
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = os.listdir(join(path, 'DATASET/counter/{}'.format(token)))
    for ii in dirlis:
        if ii.strip('.csv.gz') == book_number:
            #-open csv file and extract content
            trac_x = random.sample(list(pd.read_csv(join(path+'/DATASET/counter/{}'.format(token), ii))['word']), int(topic))
            result = ', '.join(trac_x)
    return result


@app.callback(
        Output('bar_plot', 'figure'),
        [Input('scatter_plot', 'hoverData'),
         Input('Sort-Tags', 'value'),
         Input('tokens', 'value')]
        )
def bar_plot(hoverData, sort, token):
    #--locate book and extract data from drive
    book_number = hoverData['points'][0]['customdata'][0]
    #--set extract directory
    dirlis = os.listdir(join(path, 'DATASET/counter/{}'.format(token)))
    for ii in dirlis:
        if ii.strip('.csv.gz') == book_number:
            #-open csv file and extract content
            trac_x = list(pd.read_csv(join(path+'/DATASET/counter/{}'.format(token), ii))['word'])[:15]
            trac_y = list(pd.read_csv(join(path+'/DATASET/counter/{}'.format(token), ii))['count'])[:15]
            if sort == 'A-z':
                trace = go.Bar(
                        x = trac_y,
                        y = trac_x,
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
                            yaxis = {'categoryorder': 'array',
                                     'categoryarray': [x[0].lower() for x in sorted(zip(trac_x, trac_y))],
                                     'autorange': 'reversed'}
                            )
                            
                    }
            else:
                trace = go.Bar(
                        x = trac_y,
                        y = trac_x,
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
                                yaxis = {'autorange': 'reversed'}
                                )
                                
                        }


if __name__ == '__main__':
  app.run_server(debug = True)





