import sys
from pyspark.sql import SparkSession, functions, types, Row


spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-subset'

spark.sparkContext.setLogLevel('WARN')


assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main():
    reddit_comments = spark.read.json(reddit_comments_path, schema=comments_schema)

    subs = ['Vancouver', 'movies', 'Toronto', 'politics', 'knitting'] 
    years = [2019,2024]
    months= [4,9]
    
    #filtering by time and subreddit
    filtered_data = reddit_comments.filter((reddit_comments['subreddit'].isin(subs)) &
                                           (reddit_comments['year'].isin(years))&
                                           (reddit_comments['month'].isin(months))
                                          )
    #preliminary cleaning 
    cleaned_data = filtered_data.filter((filtered_data['author']!= 'AutoModerator') &
                                          (~filtered_data['body'].isin(["[deleted]", "[removed]"]))&
                                          (~filtered_data['body'].rlike(r'http\S+'))
                                          )
    
    #Writing out filtered comments
    reddit_comments = cleaned_data.select(reddit_comments['author'], reddit_comments['body'], reddit_comments['controversiality'],
                           reddit_comments['downs'], reddit_comments['ups'], reddit_comments['year'], reddit_comments['month'],
                           reddit_comments['subreddit'], reddit_comments['score'])
    reddit_comments.show()
    reddit_comments.write.json(output + '/comments', mode='overwrite', compression='gzip')
    
    
main()
