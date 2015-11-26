package se.uu.farmbio.cp

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.commons.lang.NotImplementedException

abstract class UnderlyingAlgorithm(
  val predictor: (Vector => Double)) extends Serializable {
  def nonConformityMeasure(newSample: LabeledPoint): Double
}

abstract class UnderlyingAlgorithmSerializer[T <: UnderlyingAlgorithm] {
  //To be implemented by the user
  def serialize(alg: T): String
  def deserialize(alg: String): T
}
