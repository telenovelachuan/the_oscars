A data science project on the Shakespeare player lines dataset that aims to apply visualization and storytelling on topic modeling.

# Dataset overview

| Dataline  | Play  | Play_length  | PlayerLinenumber  | ActSceneLine  | ASL_1  | ASL_2  | ASL_3  | Player  | PL_length  | PL_w_count  | PL_w_density  | PlayerLine  | PL_contain_!  | PL_contain_?  | PL_num_comma_split  | PL_num_stop_words  | PL_num_upper_case  | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 4  | Henry IV  | 8  | 1.0  | 1.1.1  | 1.0  | 1.0  | 1.0  | KING HENRY IV  | 38  | 9  | 4.222222  | So shaken as we are, so wan with care,  | False  | False  | 3  | 5  | 1  |
| 5  | Henry IV  | 8  | 1.0  | 1.1.2  | 1.0  | 1.0  | 2.0  | KING HENRY IV  | 42  | 9  | 4.666667  | Find we a time for frighted peace to pant,  | False  | False  | 2  | 4  | 1  |

# Visualization of features

- Top 15 words in title (excluding stop words)
![top_15_words_in_title](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/feature_visualization/top_15_words_in_title.png)

- Top 15 players by frequency
![top_50_players_by_freq](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/feature_visualization/top_50_players_by_frq.png)

- Player line length distribution
![player_line_length_distr](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/feature_visualization/player_line_distr.png)

- Word cloud for all player lines
![word_count_all](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/feature_visualization/word_cloud_all_words.png)

- Word cloud for Hamlet subset
![word_count_hamlet](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/feature_visualization/word_cloud_Hamlet.png)


# Topic storytelling

Build LDA model to group topics for the Shakespeare player line dataset. Use several methodologies in dimensionality reduction and visualization for storytelling.

- Construct vectorizer, document term matrix and topic matrix for the Hamlet subset.
- use t-SNE to reduce the dimensionality of topic matrix vector into 2, and visualize
![tSNE_hamlet](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/topic_modeling/tSNE_2D_Hamlet.png)

- use pyLDAvis to do interactive visualization of topic matrix
![pyLDAvis_screenshot](https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/topic_modeling/pyLADvis_GLOUCESTER_tSNE.png)


- 3D interactive visualization of the topic matrix on Hamlet subset:
[Click Me](https://htmlpreview.github.io/?https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/topic_modeling/3D_plot_of_topics_by_LDA_on_Hamlet_subset_3D.html)

- interactive visualization by pyLDAvis to visualize the generated topic groups in Hamlet subset using PCA:
[Click Me](https://htmlpreview.github.io/?https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/topic_modeling/Hamlet_PCA_pyLDAvis.html)

- interactive visualization by pyLDAvis to visualize the generated topic groups in Hamlet subset using tSNE:
[Click Me](https://htmlpreview.github.io/?https://github.com/telenovelachuan/the_oscars/blob/master/reports/figures/topic_modeling/hamlet_tsne_pyLDAvis.html)



