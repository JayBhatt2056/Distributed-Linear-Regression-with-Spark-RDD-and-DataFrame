// Databricks notebook source
// MAGIC %md
// MAGIC **Main Project (100 pts)** \
// MAGIC Implement closed-form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X of type RDD\[Int, Int, Double\] and vector y of type RDD\[Int, Double\]
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.util.Random
import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.inv


val spark = SparkSession.builder()
  .appName("MatrixTransposeExample")
  .master("local[*]")
  .getOrCreate()


// Create a Big matrix with random values
val numRows = 100
val numCols = 10
val yNum = 3

val X = Array.fill(numRows, numCols)(Random.nextDouble())
//val Y : Array[Double] = Array.fill(Examples)(scala.util.Random.nextDouble())
//val Y = Array.fill(yNum)(Random.nextDouble())

//val X = Array(Array(5.0, 2.0), Array(4.0, 3.0), Array(7.0, 1.0))
val Y = Array(Array(1.0), Array(2.0))


val X_RDD = spark.sparkContext.parallelize(X.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })
val Y_RDD = spark.sparkContext.parallelize(Y.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })

def transposeMatrix(X: RDD[((Int, Int), Double)]): RDD[((Int, Int), Double)] = {
  X.map { case ((i, j), value) => ((j, i), value) }
}


val XT_RDD = transposeMatrix(X_RDD)

XT_RDD.collect().sortBy(_._1).foreach { case ((i, j), value) => 
  println(s"($i, $j): $value")
}

// COMMAND ----------

import org.apache.spark.rdd.RDD

def multiplyMatrices(M: RDD[((Int, Int), Double)], N: RDD[((Int, Int), Double)]): RDD[((Int, Int), Double)] = {
  val M_keyed = M.map { case ((i, j), value) => (j, (i, value)) } // key by column index for M
  val N_keyed = N.map { case ((i, j), value) => (i, (j, value)) } // key by row index for N
  val joined = M_keyed.join(N_keyed)
  val partialProducts = joined.map { case (_, ((i, m_val), (j, n_val))) => ((i, j), m_val * n_val) }
  val reducedProducts = partialProducts.reduceByKey(_ + _)

  reducedProducts
}

val XTX_RDD = multiplyMatrices(X_RDD, XT_RDD)

// Print matrixtx
XTX_RDD.collect().sortBy(_._1).foreach { case ((i, j), value) => 
  println(s"($i, $j): $value")
}

val XTy_RDD = multiplyMatrices(XT_RDD, Y_RDD)

// Print matriXTy
XTy_RDD.collect().sortBy(_._1).foreach { case ((i, j), value) => 
  println(s"($i, $j): $value")
}


// COMMAND ----------

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.inv


val XTX_RDD_2 = multiplyMatrices(X_RDD, transposeMatrix(X_RDD))
val XTy_RDD_2 = multiplyMatrices(transposeMatrix(X_RDD), Y_RDD)


val XTX_Matrix = new DenseMatrix[Double](2, 2, XTX_RDD_2.collect().sortBy(_._1).map(_._2))
val XTy_Matrix = new DenseMatrix[Double](2, 1, XTy_RDD_2.collect().sortBy(_._1).map(_._2))


// Compute theta
val theta_Matrix = inv(XTX_Matrix) * XTy_Matrix


println("Theta values:")
println(theta_Matrix)

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 1(10 pts)** \
// MAGIC Implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] using Spark DataFrame.  
// MAGIC
// MAGIC Note: Your queries should be in the following format:
// MAGIC \\[ \scriptsize \mathbf{spark.sql("select ... from ...")}\\]

// COMMAND ----------

import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.types._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, inv}


val spark = SparkSession.builder().appName("LinearRegression").getOrCreate()


val XData = Seq(
  (1, 1.0, 2.0),
  (2, 2.0, 3.0),
  (3, 3.0, 4.0)
).map { case (id, x1, x2) => Row(id, x1, x2) }


val yData = Seq(
  (1, 5.0),
  (2, 7.0),
  (3, 6.0)
).map { case (id, y) => Row(id, y) }

// Define schema for X and y (row and id)
val XSchema = StructType(Array(
  StructField("id", IntegerType, nullable = false),
  StructField("x1", DoubleType, nullable = false),
  StructField("x2", DoubleType, nullable = false)
))

val ySchema = StructType(Array(
  StructField("id", IntegerType, nullable = false),
  StructField("y", DoubleType, nullable = false)
))

val XDF = spark.createDataFrame(spark.sparkContext.parallelize(XData), XSchema)
val yDF = spark.createDataFrame(spark.sparkContext.parallelize(yData), ySchema)

XDF.createOrReplaceTempView("X")
yDF.createOrReplaceTempView("y")




// COMMAND ----------

// Compute X^T X using SQL
val XTXDF = spark.sql("""
  SELECT
    SUM(x1 * x1) as x1_x1,
    SUM(x1 * x2) as x1_x2,
    SUM(x2 * x2) as x2_x2
  FROM X
""")

XTXDF.show()


// COMMAND ----------


val XTyDF = spark.sql("""
  SELECT
    SUM(X.x1 * y.y) as x1_y,
    SUM(X.x2 * y.y) as x2_y
  FROM X
  JOIN y ON X.id = y.id
""")

XTyDF.show()

// COMMAND ----------


val XTX = XTXDF.collect().map(row => Array(row.getDouble(0), row.getDouble(1), row.getDouble(2)))
val XTy = XTyDF.collect().map(row => row.getDouble(0) +: row.getDouble(1) +: Nil)

val XTXMatrix = BDM((XTX(0)(0), XTX(0)(1)), (XTX(0)(1), XTX(0)(2)))
val XTyVector = BDV(XTy(0): _*)

val invXTX = inv(XTXMatrix)

val theta = invXTX * XTyVector

println("Theta values:")
println(theta)


// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 2(10 pts)** \
// MAGIC Run both of your implementations (main project using RDDs, bonus 1 using Dataframes) on Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download. Which implementation performs better?
// MAGIC
// MAGIC code partially done
// MAGIC

// COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.split

val spark = SparkSession.builder()
  .appName("Boston Housing")
  .getOrCreate()



val data = spark.read
  .option("delimiter", "  ")
  .option("header", "false")  
  .csv("/FileStore/tables/housing.csv")


val dataSplit = data.withColumn("_tmp", split($"_c10", " "))
                    .withColumn("_c10", $"_tmp".getItem(0))
                    .withColumn("_c10,2", $"_tmp".getItem(1))
                    .drop("_tmp")

dataSplit.show(5)
dataSplit.printSchema()


// COMMAND ----------

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix}
import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector, inv}

def computeXTX(X: RDD[MatrixEntry]): BreezeDenseMatrix[Double] = {
  val X_pairs = X.map(entry => (entry.j, (entry.i, entry.value)))
  val groupedX = X_pairs.groupByKey().collectAsMap()

  val numCols = X.map(_.j).max() + 1
  val XTX = BreezeDenseMatrix.zeros[Double](numCols.toInt, numCols.toInt)

  for {
    col1 <- groupedX.keySet.toList.sorted
    col2 <- groupedX.keySet.toList.sorted
    if col1 <= col2
    (row1, value1) <- groupedX(col1.toLong)
    (row2, value2) <- groupedX(col2.toLong)
    if row1 <= row2
  } {
    XTX(col1.toInt, col2.toInt) += value1 * value2
    if (col1 != col2) {
      XTX(col2.toInt, col1.toInt) += value1 * value2
    }
  }

  XTX
}

def computeXTy(X: RDD[MatrixEntry], y: Array[Double]): BreezeDenseVector[Double] = {
  val y_vec = BreezeDenseVector(y)
  val XTX_local = new CoordinateMatrix(X).transpose.toBlockMatrix.toLocalMatrix

  val XTy_vector = XTX_local.rowIter.toSeq
    .map(row => row.toArray.zip(y_vec.toArray).map { case (a, b) => a * b }.sum)
    .toArray

  BreezeDenseVector(XTy_vector)
}

// COMMAND ----------

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import breeze.linalg.pinv


val dataTyped = dataSplit.select(
  col("_c0").cast(DoubleType),
  col("_c1").cast(DoubleType),
  col("_c2").cast(DoubleType),
  col("_c3").cast(IntegerType),
  col("_c4").cast(DoubleType),
  col("_c5").cast(DoubleType),
  col("_c6").cast(DoubleType),
  col("_c7").cast(DoubleType),
  col("_c8").cast(IntegerType),
  col("_c9").cast(DoubleType),
  col("_c10").cast(DoubleType),  
  col("_c11").cast(DoubleType),  
  col("_c12").cast(DoubleType),
  col("_c10,2").cast(DoubleType)
).na.drop()  

def dataFrameToMatrixEntry(df: DataFrame): RDD[MatrixEntry] = {
  df.rdd.map(row => MatrixEntry(
    row.getAs[Double]("_c0").toLong,
    row.getAs[Double]("_c1").toLong,
    row.getAs[Double]("_c2")
  ))
}
val X_RDD = dataFrameToMatrixEntry(dataTyped)

// Compute X^T X
val XTX_matrix = computeXTX(X_RDD)
val y = dataTyped.select(dataTyped("_c12").cast(DoubleType)).rdd.map(_.getDouble(0)).collect()

// Compute X^T y
val XTy_vector = computeXTy(X_RDD, y)

// Compute the pseudo-inverse of X^T X
val XTX_pinv = pinv(XTX_matrix)

// Compute theta
val theta = XTX_pinv * XTy_vector
println("Theta values:")
println(theta)


// COMMAND ----------

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val spark = SparkSession.builder().appName("LinearRegressionSQL").getOrCreate()

val dataTyped = dataSplit.select(
  col("_c0").cast(DoubleType).as("feature1"),
  col("_c1").cast(DoubleType).as("feature2"),
  col("_c2").cast(DoubleType).as("feature3"),
  col("_c3").cast(IntegerType).as("feature4"),
  col("_c4").cast(DoubleType).as("feature5"),
  col("_c5").cast(DoubleType).as("feature6"),
  col("_c6").cast(DoubleType).as("feature7"),
  col("_c7").cast(DoubleType).as("feature8"),
  col("_c8").cast(IntegerType).as("feature9"),
  col("_c9").cast(DoubleType).as("feature10"),
  col("_c10").cast(DoubleType).as("feature11"),  
  col("_c11").cast(DoubleType).as("feature12"),  
  col("_c12").cast(DoubleType).as("label"),
  col("_c10,2").cast(DoubleType).as("feature13") // Correct column name if necessary
).na.drop()


dataTyped.createOrReplaceTempView("data")

val XTXDF = spark.sql("""
  SELECT 
    SUM(feature1 * feature1) as f1_f1,
    SUM(feature1 * feature2) as f1_f2,
    SUM(feature1 * feature3) as f1_f3,
    SUM(feature2 * feature2) as f2_f2,
    SUM(feature2 * feature3) as f2_f3,
    SUM(feature3 * feature3) as f3_f3
  FROM data
""")

XTXDF.show()

val XTyDF = spark.sql("""
  SELECT 
    SUM(feature1 * label) as f1_y,
    SUM(feature2 * label) as f2_y,
    SUM(feature3 * label) as f3_y
  FROM data
""")

XTyDF.show()

// COMMAND ----------

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import breeze.linalg.{DenseMatrix, DenseVector, inv}

val spark = SparkSession.builder().appName("LinearRegressionSQL").getOrCreate()


val dataTyped = dataSplit.select(
  col("_c0").cast(DoubleType).as("feature1"),
  col("_c1").cast(DoubleType).as("feature2"),
  col("_c2").cast(DoubleType).as("feature3"),
  col("_c12").cast(DoubleType).as("label")
).na.drop()

dataTyped.createOrReplaceTempView("data")

val XTXDF = spark.sql("""
  SELECT 
    SUM(feature1 * feature1) as f1_f1,
    SUM(feature1 * feature2) as f1_f2,
    SUM(feature1 * feature3) as f1_f3,
    SUM(feature2 * feature2) as f2_f2,
    SUM(feature2 * feature3) as f2_f3,
    SUM(feature3 * feature3) as f3_f3
  FROM data
""")

XTXDF.show()


val XTyDF = spark.sql("""
  SELECT 
    SUM(feature1 * label) as f1_y,
    SUM(feature2 * label) as f2_y,
    SUM(feature3 * label) as f3_y
  FROM data
""")

XTyDF.show()

val XTX = XTXDF.collect().flatMap(row => Seq(
  row.getDouble(0), row.getDouble(1), row.getDouble(2),
  row.getDouble(1), row.getDouble(3), row.getDouble(4),
  row.getDouble(2), row.getDouble(4), row.getDouble(5)
))

val XTy = XTyDF.collect().flatMap(row => Seq(
  row.getDouble(0), row.getDouble(1), row.getDouble(2)
))


val XTXMatrix = DenseMatrix.create(3, 3, XTX) // Adjust dimensions based on the number of features
val XTyVector = DenseVector(XTy)

val XTX_pinv = pinv(XTXMatrix)

val theta = XTX_pinv * XTyVector

println("Theta values:")
println(theta)

