package se.uu.farmbio.cp

import scala.util.Random

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object TestUtils {

  def generate4ClassesData(instances: Int, seed: Long): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    Seq.fill(instances)((rnd.nextInt(100), rnd.nextInt(100))).map(r => {
      val label = if (r._1 < 50 && r._2 < 50) {
        0.0
      } else if (r._1 < 50) {
        1.0
      } else if (r._2 < 50) {
        2.0
      } else {
        3.0
      }
      new LabeledPoint(label, Vectors.dense(Array(r._1.toDouble, r._2.toDouble)))
    })
  }

  def generate4ClassesTrainCalibTest(significance: Double) = {
    val numClasses = 4
    val calibSamples = 4 * numClasses * (1 / significance - 1).ceil.toInt //4 times the minimum
    val training = generate4ClassesData(instances = 80,
      seed = Random.nextLong)
    val test = generate4ClassesData(instances = 20,
      seed = Random.nextLong)
    val calibration = generate4ClassesData(instances = calibSamples,
      seed = Random.nextLong)
      .toArray
    (training, calibration, test)
  }

  def generateBinaryData(instances: Int, seed: Long): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    Seq.fill(instances)(rnd.nextInt(100)).map(r => {
      val label = if (r < 50) {
        0.0
      } else {
        1.0
      }
      new LabeledPoint(label, Vectors.dense(r))
    })
  }

  def testPerformance[T <: UnderlyingAlgorithm](
    model: ICPClassifierModel[T],
    test: RDD[LabeledPoint],
    sig: Double = 0.2,
    minEff: Double = 0.6,
    minRec: Double = 0.6) = {
    val pvAndLab = test.map { p =>
      (model.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(pvAndLab, Array(sig))
    val eff = metrics.efficiencyBySignificance(sig)
    val rec = metrics.recallBySignificance(sig)
    eff >= minEff && rec >= minRec
  }

}