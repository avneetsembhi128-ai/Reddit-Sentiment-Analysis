import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from scipy.stats import levene, normaltest

sys.path.append('/home/aks63/.local/lib/python3.10/site-packages')
spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-subset'

spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.data.path.append('/home/aks63/nltk_data')

sid = SentimentIntensityAnalyzer()

comments_schema = types.StructType([
    types.StructField('author', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('downs', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

spark = SparkSession.builder.appName('reddit_vader').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sys.path.append('/home/aks63/.local/lib/python3.10/site-packages')


def vader_sentiment(text):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.data.path.append('/home/aks63/nltk_data')

    sid = SentimentIntensityAnalyzer()
    return float(sid.polarity_scores(text)['compound'])

def main(in_directory,):
    df = spark.read.json(in_directory, schema=comments_schema) 

    vader_udf = udf(vader_sentiment, FloatType())
    df = df.withColumn('vader_score', vader_udf(df['body'])).cache()
    df_politics = df.filter(df['subreddit'] == 'politics').sample(fraction=0.002, seed=42).cache()
    df_movies = df.filter((df['subreddit'] == 'movies')&(df['year']== 2019)).sample(fraction=0.06, seed=42).cache()
    df_knitting = df.filter((df['subreddit'] == 'knitting')&(df['year']==2019)).limit(5000).sample(fraction=0.12, seed=42).cache()

    df_politics.write.json('reddit-sample/politics1', mode='overwrite', compression='gzip')
    df_movies.write.json('reddit-sample/movies1', mode='overwrite', compression='gzip')
    df_knitting.write.json('reddit-sample/knitting1', mode='overwrite', compression='gzip')
    
    #num_rows = df.count() #: 6 million


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
