from pyspark.mllib.linalg import Vectors

from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.param import Param, Params
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import DecisionTree
from pyspark.ml import Pipeline
sc = SparkContext()
spark = SparkSession(sc)


inputDF = spark.read.csv('s3://himaniproject2/TrainingDataset.csv',header='true', inferSchema='true', sep=';')


featureColumns = [c for c in inputDF.columns if c != 'quality']

transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))



model = DecisionTree.trainClassifier(transformed_df,numClasses=10,categoricalFeaturesInfo={}, impurity='gini', maxDepth=4,maxBins=200)


model.save(sc,"s3://himaniproject2/model")

