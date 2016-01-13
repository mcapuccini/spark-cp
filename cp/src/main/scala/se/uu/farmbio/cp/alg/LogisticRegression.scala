package se.uu.farmbio.cp.alg

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import se.uu.farmbio.cp.UnderlyingAlgorithm

//Define a LogisticRegression UnderlyingAlgorithm
private object LogisticRegression {
  def trainingProcedure(
    input: RDD[LabeledPoint],
    maxNumItearations: Int,
    regParam: Double,
    numCorrections: Int,
    convergenceTol: Double): (Vector => Double) = {

    //Train Logistic Regression with LBFGS
    val numFeatures = input.take(1)(0).features.size
    val training = input.map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
    val (weightsWithIntercept, _) = LBFGS.runLBFGS(
      training,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumItearations,
      regParam,
      initialWeightsWithIntercept)

    //Create the model using the weights
    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))

    //Return raw score predictor
    model.clearThreshold()
    model.predict

  }
}

class LogisticRegression(
  private val input: RDD[LabeledPoint],
  private val maxNumItearations: Int = 100,
  private val regParam: Double = 0.1,
  private val numCorrections: Int = 10,
  private val convergenceTol: Double = 1e-4)
  extends UnderlyingAlgorithm(
    LogisticRegression.trainingProcedure(
      input,
      maxNumItearations,
      regParam,
      numCorrections,
      convergenceTol)) {
  override def nonConformityMeasure(newSample: LabeledPoint) = {
    val score = predictor(newSample.features)
    if (newSample.label == 1.0) {
      1-score
    } else {
      score
    }
  }
}