# NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY ![](https://img.shields.io/badge/python-v3.6-orange.svg)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Project is hosted live on Heroku [![Hosted Live](https://img.shields.io/badge/Hosted-Live-brightgreen.svg?style=flat)](https://bkinsight.herokuapp.com/)

Project implements machine learning model for Natural Language Processing (NLP).
Visualization is done with Plotly Dash.
Flexibility of hovering over data points to visualize book properties (meta-data) and similarity score, 
horizontal bar chart and book imprint.
Major processing on books to extract tokenized and lemmatized features, principal component analysis for dimension reduction,
and Kmeans clustering to visualize relationship among books.
Project is hosted live on heroku.


## PROJECT WORKFLOW

<ul>
  <li>Import and and preprocess all 148 French books</li>
  <li>Stemming & Lemmatization of extracted tokens</li>
  <li>Visualize most frequent words on hover. Return ordered Barplot</li>
  <li>TF-IDF Model</li>
  
  <li>Document Similarity using Cosine distance of book content</li>
    <ul>Principal component analysis</ul>
    <ul>KMeans clustering</ul>
  <li><s>Topic Models</s></li>
    <ul><s>LatentDirichletAllocation</s></ul>
</ul>

## HOW TO USE

```python
git clone https://github.com/kennedyCzar/NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY
```
Open the script folder in your terminal and run the following command

```python
python mplot_script.py
```

```python
Navigate http://127.0.0.1:8050/ 
```
![Image 1](https://github.com/kennedyCzar/NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY/blob/master/bkinsight.gif)
