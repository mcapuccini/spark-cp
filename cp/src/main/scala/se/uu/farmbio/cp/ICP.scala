package se.uu.farmbio.cp

import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object ICP extends Logging {

  private def sampleCalibrationAndTraining(input: RDD[LabeledPoint],
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

  def splitCalibrationAndTraining(input: RDD[LabeledPoint], numOfCalibSamples: Int,
                                  bothClasses: Boolean = false) = {
    if (bothClasses) {
      //bothClasses works only for binary classification at the moment
      val sc = input.context
      val class0 = input.filter(_.label == 0.0)
      val class1 = input.filter(_.label == 1.0)
      val (calibSet0, propTraining0) = sampleCalibrationAndTraining(class0, numOfCalibSamples)
      val (calibSet1, propTraining1) = sampleCalibrationAndTraining(class1, numOfCalibSamples)
      (calibSet0 ++ calibSet1, sc.union(propTraining0, propTraining1))
    } else {
      sampleCalibrationAndTraining(input, numOfCalibSamples)
    }
  }

  def trainClassifier[A<:UnderlyingAlgorithm](
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