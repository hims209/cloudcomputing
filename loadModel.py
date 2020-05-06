from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint

from pyspark.ml.param import Param, Params
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTreeModel
sc = SparkContext()
spark = SparkSession(sc)

inputDF = spark.read.csv('s3://himaniproject2/ValidationDataset.csv',header='true', inferSchema='true', sep=';')

transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))


model = DecisionTreeModel.load(sc,"s3://himaniproject2/model")


predictions = model.predict(transformed_df.map(lambda x: x.features))

labels_and_predictions = transformed_df.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(transformed_df.count())
print(".........................................................")
print("Model accuracy....................: %.3f%%" % (acc * 100))


metrics = MulticlassMetrics(labels_and_predictions)

fscore = metrics.fMeasure()
print(".........................................................")
print("F1 Score.................................. = %s" % fscore)
