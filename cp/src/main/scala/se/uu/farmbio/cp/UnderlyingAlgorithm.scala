package se.uu.farmbio.cp

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

abstract class UnderlyingAlgorithm(
  val predictor: (Vector => Double)) extends Serializable {
  def nonConformityMeasure(newSample: LabeledPoint): Double
}

trait Deserializer[A <: UnderlyingAlgorithm] {
  def deserialize(alg: String): A
} 
