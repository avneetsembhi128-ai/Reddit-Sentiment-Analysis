
import matplotlib.pyplot as plt
import seaborn as sns

def plot_groups(data, name, title):

    sns.set_theme(style="whitegrid")
    plot = sns.kdeplot(
        data=data,
        x="vader_score",
        hue="subreddit",
        fill=True,
        alpha=0.2,
        linewidth=1.5
    )

    plot.set_xlabel("Sentiment Score")
    plot.set_ylabel("Density")
    plot.set_title(title)

    handles, labels = plot.get_legend_handles_labels()
    if handles:
        plot.legend(handles=handles, labels=labels, title="Subreddit")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def box_plot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='subreddit', y='vader_score', palette='Set2')
    plt.title('Distribution of Sentiment by Subreddit', fontsize=14)
    plt.xlabel('Subreddit', fontsize=12)
    plt.ylabel('Vader Score', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('score_by_subreddit')


def scatter_plot(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='vader_score', y='ups', hue='subreddit', alpha=0.6, palette='Set1')
    plt.title('Relationship Between Sentiment and Upvotes', fontsize=14)
    plt.xlabel('VADER Score', fontsize=12)
    plt.ylabel('Upvotes', fontsize=12)
    plt.legend(title='Subreddit')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('sentiment_vs_ups')
