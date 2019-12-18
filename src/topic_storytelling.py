import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot


SELF_DEFINED_STOP_WORDS = ['thy', 'thou', 'thee', 'sir', 'shall']
TOPIC_COLOR_MAP = np.array([
    "#1f77b5", "#aec7e9", "#ff7f0f", "#ffbb79", "#2ca02d", "#98df8b", "#d62729", "#ff9897", "#9467be", "#c5b0d6",
    "#8c564c", "#c49c95", "#e377c3", "#f7b6d3", "#7f7f80", "#c7c7c8", "#bcbd23", "#dbdb8e", "#17bed0", "#9edae6"
])


def get_keys(topic_matrix):
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def _get_mean_topic_vectors(keys, two_dim_vectors):
    '''
    returns a list of centroid vectors from each predicted topic category
    '''
    centroid_topic_vectors = []
    for t in range(n_components):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])

        articles_in_that_topic = np.vstack(articles_in_that_topic)
        centroid_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        centroid_topic_vectors.append(centroid_article_in_that_topic)
    return centroid_topic_vectors


def get_top_n_words(n, n_topics, document_term_matrix, keys, count_vectorizer):
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words

df = pd.read_csv("../data/processed/processed.csv")
df.dropna(subset=['PlayerLine'], inplace=True)
print "Load the processed data of the Shakespear playerline data set"


playerlines_entire = df['PlayerLine']
count_vectorizer_entire = CountVectorizer(stop_words='english', max_features=40000)
document_term_matrix_entire = count_vectorizer_entire.fit_transform(playerlines_entire)
print "term matrix generated"
LDA_model_entire = LatentDirichletAllocation(learning_method='online', n_components=14, random_state=0, verbose=0)
print "LDA model initialized"
topic_matrix_entire = LDA_model_entire.fit_transform(document_term_matrix_entire)
print "LDA model trained"


n_components = 14
tsne_model_entire = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_entire = tsne_model_entire.fit_transform(topic_matrix_entire)


df_hamplet = df[df['Play'] == 'Hamlet']
playerlines_hamlet = df_hamplet['PlayerLine']
count_vectorizer_hamlet = CountVectorizer(stop_words='english', max_features=40000)
document_term_matrix_hamlet = count_vectorizer_hamlet.fit_transform(playerlines_hamlet)
print "term matrix generated"
LDA_model_hamlet = LatentDirichletAllocation(learning_method='online', n_components=14, random_state=0, verbose=0)
print "LDA model initialized"
topic_matrix_hamlet = LDA_model_hamlet.fit_transform(document_term_matrix_hamlet)
print "LDA model trained"

n_components = 14
tsne_model_hamlet = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_hamlet = tsne_model_hamlet.fit_transform(topic_matrix_hamlet)

keys_hamlet = get_keys(topic_matrix_hamlet)
mean_topic_vectors_hamlet = _get_mean_topic_vectors(keys_hamlet, tsne_vectors_hamlet)

colormap = TOPIC_COLOR_MAP[:n_components]
top_3_words_hamlet = get_top_n_words(3, n_components, document_term_matrix_hamlet, keys_hamlet, count_vectorizer_hamlet)
fig, ax = plt.subplots(figsize=(16, 16))
plt.scatter(x=tsne_vectors_hamlet[:, 0], y=tsne_vectors_hamlet[:, 1],color=colormap[keys], marker='o', alpha=0.5)
for t in range(n_components):
    plt.text(mean_topic_vectors_hamlet[t][0], mean_topic_vectors_hamlet[t][1], top_3_words_hamlet[t], color=colormap[t], horizontalalignment='center', weight='bold')
plt.show()

tsne_3d_model_hamlet = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_3d_vectors_hamlet = tsne_3d_model.fit_transform(topic_matrix_hamlet)


def plot_3d_interactive(tsne_3d_vectors, title=None):
    df_tsne_vectors = pd.DataFrame(tsne_3d_vectors)

    from plotly.graph_objs import *
    init_notebook_mode()
    colormap = TOPIC_COLOR_MAP[:n_components]
    trace0 = Scatter3d(x=df_tsne_vectors[0], y=df_tsne_vectors[1], z=df_tsne_vectors[2], mode='markers',
                       marker=dict(
                           size=2,
                           color=colormap[keys],
                           # colorscale=[[0, anomaly_color], [0.5, anomaly_color], [1.0, normal_color]],
                           symbol='circle',
                           opacity=0.5
                       )
                       )
    data = [trace0]
    layout = Layout(showlegend=False, height=800, width=800)
    fig = dict(data=data, layout=layout)
    plot_title = "3D plot by plotly"
    if title:
        plot_title = "_".join(title.split(" "))
    import plotly
    plotly.offline.plot(fig, filename='../reports/figures/topic_modeling/{}_3D.html'.format(plot_title))
    iplot(fig)


plot_3d_interactive(tsne_3d_vectors_hamlet, "3D plot of topics by LDA on Hamlet subset")

df_GLOUCESTER = df[df['Player'] == 'GLOUCESTER']
n_components = 10;
playerlines = df_GLOUCESTER['PlayerLine']
count_vectorizer_GLOUCESTER = CountVectorizer(stop_words='english', max_features=40000)
document_term_matrix_GLOUCESTER = count_vectorizer_GLOUCESTER.fit_transform(playerlines)
print "term matrix generated"
LDA_model_GLOUCESTER = LatentDirichletAllocation(learning_method='online', n_components=n_components, random_state=0, verbose=0)
print "LDA model initialized"
topic_matrix_GLOUCESTER = LDA_model_GLOUCESTER.fit_transform(document_term_matrix_GLOUCESTER)
print "LDA model trained"


tsne_2d_model_GLOUCESTER = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_2d_vectors_GLOUCESTER = tsne_2d_model_GLOUCESTER.fit_transform(topic_matrix_GLOUCESTER)
keys = get_keys(topic_matrix_GLOUCESTER)
mean_topic_vectors_GLOUCESTER = _get_mean_topic_vectors(keys, tsne_2d_vectors_GLOUCESTER)
top_3_words_GLOUCESTER = get_top_n_words(3, n_components, document_term_matrix_GLOUCESTER, keys, count_vectorizer_GLOUCESTER)
colormap = TOPIC_COLOR_MAP[:n_components]
fig, ax = plt.subplots(figsize=(16, 16))
plt.scatter(x=tsne_2d_vectors_GLOUCESTER[:, 0], y=tsne_2d_vectors_GLOUCESTER[:, 1],color=colormap[keys],marker='o', alpha=0.5)
for t in range(n_components):
    plt.text(mean_topic_vectors_GLOUCESTER[t][0], mean_topic_vectors_GLOUCESTER[t][1], top_3_words_GLOUCESTER[t], color=colormap[t],  horizontalalignment='center', weight='bold')
plt.show()

tsne_3d_model_GLOUCESTER = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_3d_vectors_GLOUCESTER = tsne_3d_model_GLOUCESTER.fit_transform(topic_matrix_GLOUCESTER)

plot_3d_interactive(tsne_3d_vectors_GLOUCESTER, "3D plot of topics in GLOUCESTER subset by LDA")

from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
hamlet_prepared_PCA = pyLDAvis.sklearn.prepare(LDA_model_hamlet, document_term_matrix_hamlet, count_vectorizer_hamlet)
pyLDAvis.save_html(hamlet_prepared, '../reports/figures/topic_modeling/{}_pyLDAvis.html'.format("Hamlet_PCA"))
pyLDAvis.display(hamlet_prepared_PCA)


hamlet_mmds_prepared = pyLDAvis.sklearn.prepare(LDA_model_hamlet, document_term_matrix_hamlet, count_vectorizer_hamlet, mds='mmds')
pyLDAvis.save_html(hamlet_mmds_prepared, '../reports/figures/topic_modeling/{}_pyLDAvis.html'.format('Hamlet_mmds'))
pyLDAvis.display(hamlet_mmds_prepared)


hamlet_tsne_prepared = pyLDAvis.sklearn.prepare(LDA_model_hamlet, document_term_matrix_hamlet, count_vectorizer_hamlet, mds='tsne')
pyLDAvis.save_html(hamlet_tsne_prepared, '../reports/figures/topic_modeling/{}_pyLDAvis.html'.format('hamlet_tsne'))
pyLDAvis.display(hamlet_tsne_prepared)

GLOUCESTER_pca_prepared = pyLDAvis.sklearn.prepare(LDA_model_GLOUCESTER, document_term_matrix_GLOUCESTER, count_vectorizer_GLOUCESTER, mds='pcoa')
pyLDAvis.save_html(GLOUCESTER_pca_prepared, '../reports/figures/topic_modeling/{}_pyLDAvis.html'.format('GLOUCESTER_pca'))
pyLDAvis.display(GLOUCESTER_pca_prepared)


GLOUCESTER_tsne_prepared = pyLDAvis.sklearn.prepare(LDA_model_GLOUCESTER, document_term_matrix_GLOUCESTER, count_vectorizer_GLOUCESTER, mds='tsne')
pyLDAvis.save_html(GLOUCESTER_tsne_prepared, '../reports/figures/topic_modeling/{}_pyLDAvis.html'.format('GLOUCESTER_tsne'))
pyLDAvis.display(GLOUCESTER_tsne_prepared)




