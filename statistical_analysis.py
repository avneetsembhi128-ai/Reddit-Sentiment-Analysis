import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from Plots import plot_groups, box_plot, scatter_plot


files = [
    "politics1.json",
   "knitting1.json",
    "movies1.json",
    "politics2.json",
   "knitting2.json",
    "movies2.json"
]

dfs = [pd.read_json(f, lines=True) for f in files]
reddit_df = pd.concat(dfs, ignore_index=True)

#plot sentiment distribution for the three groups
plot_groups(reddit_df, 'general_sentiment.png', "Sentiment Score Distribution by Subreddit")

negative_scores = reddit_df[reddit_df['vader_score'] < 0]
past = negative_scores[negative_scores['year'] == 2019]
present = negative_scores[negative_scores['year'] == 2024]

#p<0.5 for normality and variance 
normality_past = stats.normaltest(past['vader_score']).pvalue
normality_present = stats.normaltest(present['vader_score']).pvalue
variance = stats.levene(past['vader_score'], present['vader_score']).pvalue

#p>0.05 for mannwhitneyu = no significant change in negativity over time
utest = stats.mannwhitneyu(past['vader_score'],present['vader_score']).pvalue
print("MannWhitneyU p-value (overall):", utest)

#finding which groups were more negative
knitting_scores = negative_scores[negative_scores['subreddit'] == 'knitting']['vader_score']
politics_scores = negative_scores[negative_scores['subreddit'] == 'politics']['vader_score']
movies_scores = negative_scores[negative_scores['subreddit'] == 'movies']['vader_score']

plot_groups(negative_scores, 'negativity_per_group.png', "Negative Score Distribution by Subreddit")
plot_groups(past, 'negative_scores_2019.png', "Negative Score Distribution by Subreddit 2019")
plot_groups(present, 'negative_scores_2024.png', "Negative Score Distribution by Subreddit 2024")

#p<0.05 for kruskal test 
kruskal = stats.kruskal(knitting_scores, politics_scores, movies_scores).pvalue
print("Kruskal-Wallis (p-value):", kruskal)

dunn = sp.posthoc_dunn(
    negative_scores,
    val_col='vader_score',
    group_col='subreddit',
    p_adjust='bonferroni'                  
)
print("Dunn's Test Result (p-values):")
print(dunn)

negative_movie = reddit_df[(reddit_df['subreddit'] == 'movies')&(reddit_df['ups'] < 300)]
negative_movie = negative_movie.dropna(subset=['ups'])

#spearman correlation between Vader score and upvotes for movies
spearman = stats.spearmanr(negative_movie['vader_score'], negative_movie['ups']).pvalue
print(f"Spearman correlation: {spearman:.4f}")

#plot of sentiment scores by subreddit
box_plot(negative_scores)

#median sentiment scores
print(negative_scores.groupby('subreddit')['vader_score'].median())

#plot of sentiment scores vs ups

scatter_plot(negative_movie)
