package se.uu.farmbio.cp

import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object ICP extends Logging {

  private def simpleSplit(
    input: RDD[LabeledPoint],
    numOfCalibSamples: Int) = {

    //Computing the calibration fraction using binomial upper bound
    val n = input.count
    val fraction = numOfCalibSamples.toDouble / n
    val delta = 1e-4
    val minSamplingRate = 1e-10
    val gamma = -math.log(delta) / n
    val calibFraction = math.min(1,
      math.max(minSamplingRate, fraction + gamma + math.sqrt(gamma * gamma + 2 * gamma * fraction)))

    //calibFraction is enough most of the times, but not always 
    val splits = input.randomSplit(Array(calibFraction, 1 - calibFraction))
    var sample = splits(0).collect
    while (sample.length < numOfCalibSamples) {
      logWarning("Needed to re-sample calibration set due to insufficient sample size.")
      val split = input.randomSplit(Array(calibFraction, 1 - calibFraction))
      sample = splits(0).collect
    }

    val calibration = sample.take(numOfCalibSamples)
    val additional = sample.takeRight(sample.length - numOfCalibSamples)

    val sc = input.context
    (calibration, splits(1) ++ sc.parallelize(additional))

  }

  private def stratifiedSplit(
    input: RDD[LabeledPoint],
    numOfCalibSamples: Int) = {

    logWarning("Stratified sampling is supported only for binary classification.")
    
    //Calibration split, making sure there is some data for both classes
    val class0 = input.filter(_.label == 0.0)
    val class1 = input.filter(_.label == 1.0)
    val count0 = class0.count
    val count1 = class1.count
    val negSize = ((count0.doubleValue / (count0 + count1)) * numOfCalibSamples)
      .ceil.toInt
    val posSize = numOfCalibSamples - negSize
    val (negSmpl, negTr) = ICP.simpleSplit(class0, negSize)
    val (posSmpl, posTr) = ICP.simpleSplit(class1, posSize)
    val properTraining = negTr ++ posTr
    val clalibration = negSmpl ++ posSmpl
    (clalibration, properTraining)

  }

  def calibrationSplit(
    input: RDD[LabeledPoint],
    numOfCalibSamples: Int,
    stratified: Boolean = false) = {

    if (stratified) {
      logWarning("Stratified sampling needs to count the dataset, you should use it wisely.")
      ICP.stratifiedSplit(input, numOfCalibSamples)
    } else {
      ICP.simpleSplit(input, numOfCalibSamples)
    }

  }

  def trainClassifier[A <: UnderlyingAlgorithm](
    alg: A,
    numClasses: Int,
    calibSet: Array[LabeledPoint]): ICPClassifierModel[A] = {
    //Compute aphas for each class (mondrian approach)
    val alphas = (0 to numClasses - 1).map { i =>
      calibSet.filter(_.label == i) //filter current label
        .map(newSmpl => alg.nonConformityMeasure(newSmpl)) //compute alpha
    }
    new ICPClassifierModelImpl(alg, alphas)
  }

}