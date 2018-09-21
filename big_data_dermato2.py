#from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
#from pyspark.mllib.util import MLUtils
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


sc=SparkContext("local[*]",appName="nom")
spark = SparkSession(sc)
# Load and parse the data file.
#data = MLUtils.loadLibSVMFile(sc, "/home/user/Bureau/spark/projet1/bigdatadermato.csv")

df = spark.read.format("csv").option("header", "true").option("sep",",").option("inferSchema","true").load("/home/user/Bureau/spark/projet1/bigdatadermato.csv")
#df.show(10)

from pyspark.ml.feature import StringIndexer
disieseIndexer=StringIndexer(inputCol="Classe",outputCol="disieseIndex")
from pyspark.ml.feature import VectorAssembler
featureColumns='\"V1\"'
for i in range(2,225):
	featureColumns=featureColumns+','+'\"V'+str(i)+'\"'

vectorAssembler=VectorAssembler(inputCols=[featureColumns],outputCol="features")
data=vectorAssembler.transform(df)
index_model=disieseIndexer.fit(data)
data_indexed=index_model.transform(data)


trainingData, testData = data_indexed.randomSplit([0.8, 0.2],0.0)

from pyspark.ml.classification import DecisionTreeClassifier
dt=DecisionTreeClassifier().setLabelCol("disieseIndex").setFeaturesCol("features")
model = dt.fit(trainingData)
classifications =model.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator=MulticlassClassificationEvaluator(labelCol="disieseIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(classifications)
print("********************************************************************************")
print("Test set accuracy = " + str(accuracy))
print("********************************************************************************")

