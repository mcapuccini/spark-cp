package se.uu.farmbio.cp.alg

import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import org.apache.spark.SharedSparkContext
import se.uu.farmbio.cp.ICPClassifierModel
import org.apache.spark.mllib.regression.LabeledPoint
import se.uu.farmbio.cp.UnderlyingAlgorithm
import org.apache.spark.rdd.RDD

object TestUtils {

  def testPerformance[T <: UnderlyingAlgorithm](
    model: ICPClassifierModel[T],
    test: RDD[LabeledPoint],
    sig: Double = 0.2,
    minEff: Double = 0.7,
    minRec: Double = 0.7) = {
    val pvAndLab = test.map { p =>
      (model.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(pvAndLab, Array(sig))
    val eff = metrics.efficiencyBySignificance(sig)
    val rec = metrics.recallBySignificance(sig)
    eff >= minEff && rec >= minRec
  }

}