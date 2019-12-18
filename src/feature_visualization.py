import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

SELF_DEFINED_STOP_WORDS = ['thy', 'thou', 'thee', 'sir', 'shall']
df = pd.read_csv("../data/processed/processed.csv")
df.dropna(subset=['PlayerLine'], inplace=True)
print "Load the processed data of the Shakespear playerline data set"


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if word not in SELF_DEFINED_STOP_WORDS]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_top_n_words(top_words, title=None, rotation=45):
    count = len(top_words)
    fig, ax = plt.subplots(figsize=(16, 8))
    words = [x[0] for x in top_words]
    ax.bar(range(count), [x[1] for x in top_words])
    ax.set_xticks(range(count))
    ax.set_xticklabels(words, rotation='vertical')
    chart_title = title or 'Top {} words in title (excluding stop words)'.format(count)
    ax.set_title(chart_title)
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=rotation)
    plt.show()


titles = df['PlayerLine'].dropna()
top_N_words = get_top_n_words(titles, n=15)
plot_top_n_words(top_N_words, rotation=0)

print "number of playerlines by player"
plt.figure(figsize=(16, 7))
lines_by_player = df['Player'].value_counts(sort=True)[:50]
lines_by_player
fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(lines_by_player.index.values, lines_by_player.values)
#ax = sns.lineplot(x=lines_by_player.index.values, y=lines_by_player.values)
plt.title('Top 50 players by frequency')
plt.xticks(rotation=45)
plt.show()


print "number of playerlines by play"
plt.figure(figsize=(16, 7))
lines_by_play = df['Play'].value_counts(sort=True)
fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(lines_by_play.index.values, lines_by_play.values)
plt.title('Play by frequency')
plt.xticks(rotation=45)
plt.show()


pl_length = df['PL_length'].value_counts(sort=True)
plt.figure(figsize=(16, 6))
plt.title("Playerline length distribution")
sns.distplot(pl_length.values);


pl_word_count = df['PL_w_count'].value_counts(sort=True)
pl_word_count[:20].index
pl_word_count_top_20 = df[df['PL_w_count'].isin(pl_word_count[:20].index)]

plt.figure(figsize=(16, 6))
plt.title("Playerline word count distribution")
sns.distplot(pl_word_count_top_20['PL_w_count'], bins=20, kde=False, hist_kws={"rwidth":0.8,"alpha": 1, "color": "lightblue"});



binwidth = 0.05
plt.figure(figsize=(16, 6))
#plt.hist(df['PL_w_density'], bins = int(10/binwidth), color = 'lightblue', edgecolor = 'black', log=True)
sns.distplot(df['PL_w_density'], bins=100, kde=True, rug=False,
             hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "y"})
plt.title("Player line word density distribution")


def plot_wordcloud(words, title):
    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white",
                      stopwords=list(STOPWORDS) + SELF_DEFINED_STOP_WORDS).generate(words)
    plt.figure(figsize=(20,20))
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off")
    plt.title(title, fontsize=60)
    plt.show()


all_text = " ".join([str(v) for v in df['PlayerLine']])
plot_wordcloud(all_text, "All playerlines")

print "What is Hamlet talking about?"
df_hamlet = df[df['Play'] == 'Hamlet']
top_11_words = get_top_n_words(df_hamlet.PlayerLine, n=11)
print top_11_words
plot_top_n_words(top_11_words, title="Top 10 words in Hamlet", rotation=0)


hamlet_text = " ".join([str(v) for v in df_hamlet['PlayerLine']])
plot_wordcloud(hamlet_text, "Hamlet playerlines")

df_tragedy = df[df['PlayerLine'].str.contains("miseries|misery|tragedy|tragedies")]
plays = df_tragedy['Play'].value_counts(sort=False).sort_index()
plt.figure(figsize=(16, 6))
rank = plays.argsort().argsort()
pal = sns.color_palette("RdBu", len(rank))
pal
sns.barplot(plays.index.values, plays.values, palette=np.array(pal[::-1])[rank]);
plt.xticks(rotation=45)
plt.title("mentions of 'tragedy/misery' in all plays")




