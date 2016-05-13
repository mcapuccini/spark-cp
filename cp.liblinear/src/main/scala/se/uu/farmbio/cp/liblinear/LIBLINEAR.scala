package se.uu.farmbio.cp.liblinear

import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint

import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.ICP

object LIBLINEAR {
  
  private def calibrationSplit(
    trainingData: Array[LabeledPoint],
    calibrationSizeP: Int,
    calibrationSizeN: Int) = {
    val shuffData = Random.shuffle(trainingData.toList)
    val positives = shuffData.filter { p => p.label == 1.0 }
    val negatives = shuffData.filter { p => p.label != 1.0 }
    val calibration = (
      positives.take(calibrationSizeP) ++
      negatives.take(calibrationSizeN))
      .toArray
    val properTraining = (
      //Negative labels go first
      negatives.takeRight(negatives.length - calibrationSizeN) ++
      positives.takeRight(positives.length - calibrationSizeP))
      .toArray
    (properTraining, calibration)
  }

  private[liblinear] def splitFractional(
    trainingData: Array[LabeledPoint],
    calibrationFraction: Double) = {
    val calibrationSizeP = (trainingData.count(_.label == 1.0) * calibrationFraction).toInt
    val calibrationSizeN = (trainingData.count(_.label != 1.0) * calibrationFraction).toInt
    calibrationSplit(trainingData, calibrationSizeP, calibrationSizeN)
  }
  
  def trainAggregatedICPClassifier(
    sc: SparkContext,
    trainingData: Array[LabeledPoint],
    calibrationFraction: Double = 0.2,
    numberOfICPs: Int = 30,
    solverType: SolverType = SolverType.L2R_L2LOSS_SVC_DUAL,
    regParam: Double = 1,
    tol: Double = 0.01) = {

    //Broadcast the dataset
    val trainBroadcast = sc.broadcast(trainingData)

    //Train ICPs for different calibration samples
    val icps = sc.parallelize((1 to numberOfICPs)).map { _ =>
      //Sample calibration
      val (properTraining, calibration) = splitFractional(trainBroadcast.value, calibrationFraction)
      //Train ICP
      val alg = new LibLinAlg(
        properTraining,
        solverType,
        regParam,
        tol)
      ICP.trainClassifier(alg, numClasses = 2, calibration)
    }
    
    new AggregatedICPClassifier(icps)
    
  }
  
}