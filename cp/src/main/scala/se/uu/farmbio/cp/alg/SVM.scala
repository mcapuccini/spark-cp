package se.uu.farmbio.cp.alg

import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.HingeGradient
import org.apache.spark.mllib.regression.LabeledPoint
import se.uu.farmbio.cp.UnderlyingAlgorithmSerializer
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMModel
import se.uu.farmbio.cp.UnderlyingAlgorithm
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.util.MLUtils

//Define a SVMs UnderlyingAlgorithm
private object SVM {
  def trainingProcedure(
    input: RDD[LabeledPoint],
    maxNumItearations: Int,
    regParam: Double,
    numCorrections: Int,
    convergenceTol: Double) = {

    //Train SVM with LBFGS
    val numFeatures = input.take(1)(0).features.size
    val training = input.map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
    val (weightsWithIntercept, _) = LBFGS.runLBFGS(
      training,
      new HingeGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumItearations,
      regParam,
      initialWeightsWithIntercept)

    //Create the model using the weights
    val model = new SVMModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))

    //Return raw score predictor
    model.clearThreshold()
    model

  }
}

class SVM(val model: SVMModel)
  extends UnderlyingAlgorithm(model.predict) {

  def this(
    input: RDD[LabeledPoint],
    maxNumItearations: Int = 100,
    regParam: Double = 0.1,
    numCorrections: Int = 10,
    convergenceTol: Double = 1e-4) = {

    this(SVM.trainingProcedure(
      input,
      maxNumItearations,
      regParam,
      numCorrections,
      convergenceTol))

  }
  
  def nonConformityMeasure(newSample: LabeledPoint) = {
    val score = predictor(newSample.features)
    if (newSample.label == 1.0) {
      -score
    } else {
      score
    }
  }
  
}

object SVMSerializer extends UnderlyingAlgorithmSerializer[SVM] {
  override def serialize(alg: SVM): String = {
    alg.model.intercept + "\n" +
      alg.model.weights.toString
  }
  override def deserialize(modelString: String): SVM = {
    val rowSplitted = modelString.split("\n")
    val intercept = rowSplitted(0)
    val weights = rowSplitted(1)
    val model = new SVMModel(Vectors.parse(weights).toSparse, intercept.toDouble)
    model.clearThreshold()
    new SVM(model)
  }
}