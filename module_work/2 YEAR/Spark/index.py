from pyspark.shell import spark

from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql.functions import split, size

df = df.withColumn("length",size(split("v2"," ")))

conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)