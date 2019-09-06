# NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY ![](https://img.shields.io/badge/python-v3.6-orange.svg)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Project is hosted live on Heroku [![Hosted Live](https://img.shields.io/badge/Hosted-Live-brightgreen.svg?style=flat)](https://bkinsight.herokuapp.com/)

Project implements machine learning model for natural language processing.
Visualization is done with Plotly Dash.
Dynamically hover over data point to display book meta data;
book title, year and author name just below the chart.
Plus it displays dynamically a barchart to see the frequency of word
in the book after some initial preprocessing.
Clustering of book similarity.
Project is hosted live on heroku.


## PROJECT WORKFLOW

<ul>
  <li>Import and and preprocess all 148 books</li>
  <li>Stemming & Lemmatization of extracted tokens</li>
  <li>Visualize most frequent words on hover. Return ordered Barplot</li>
  <li>TF-IDF Model</li>
  
  <li>Document Similarity using Cosine distance of book content</li>
    <ul>Principal component analysis</ul>
    <ul>KMeans clustering</ul>
  <li><s>Topic Models</li>
    <ul><s>LatentDirichletAllocation</ul>
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
![Image 1](https://github.com/kennedyCzar/NLP-PROJECT-BOOK-INSIGHTS-WITH-PLOTLY/blob/master/vid.gif)
