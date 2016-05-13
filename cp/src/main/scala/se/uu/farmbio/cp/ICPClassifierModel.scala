package se.uu.farmbio.cp

import java.io.Serializable
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

object ICPClassifierModel {
  
  def deserialize[A <: UnderlyingAlgorithm](
      model: String, 
      algDeserializer: Deserializer[A]): ICPClassifierModel[A] = {
    
    //If curly brackets are used in the  algorithm serialization this won't work
    val matches = "\\{(.*?)\\}".r 
      .findAllMatchIn(model)
      .map(_.matched)
      .toArray
    if(matches.length != 2) {
      throw new IllegalArgumentException("malformed model string")
    }
    val algStr = matches(0).substring(1, matches(0).length-1)
    val alg = algDeserializer.deserialize(algStr)
    val alphStr = matches(1).substring(1, matches(1).length-1)
    val alph = "\\(([-+]?[0-9]*\\.?[0-9]+,)*[-+]?[0-9]*\\.?[0-9]+\\)".r
      .findAllMatchIn(alphStr).map{ pairMatch => 
        val pairStr = pairMatch.matched
        pairStr.substring(1,pairStr.length-1)
          .split(",")
          .map(_.toDouble)
      }.toSeq
      new ICPClassifierModelImpl(alg,alph)
  }
  
}

abstract class ICPClassifierModel[A <: UnderlyingAlgorithm]
  extends Serializable {

  def mondrianPv(features: Vector): Array[Double]

  def predict(features: Vector, significance: Double) = {
    //Compute region
    mondrianPv(features).zipWithIndex.map {
      case (pVal, c) =>
        if (pVal > significance) {
          Set(c.toDouble)
        } else {
          Set[Double]()
        }
    }.reduce(_ ++ _)
  }

}

private[cp] class ICPClassifierModelImpl[A <: UnderlyingAlgorithm](
  val alg: A,
  val alphas: Seq[Array[Double]])
  extends ICPClassifierModel[A] with Logging {

  override def mondrianPv(features: Vector) = {
    (0 to alphas.length - 1).map { i =>
      //compute non-conformity for new example
      val alphaN = alg.nonConformityMeasure(new LabeledPoint(i, features))
      //compute p-value
      (alphas(i).count(_ >= alphaN) + 1).toDouble /
        (alphas(i).length.toDouble + 1)
    }.toArray
  }

  override def predict(features: Vector, significance: Double) = {
    //Validate input
    alphas.foreach { a =>
      require(significance > 0 && significance < 1, s"significance $significance is not in (0,1)")
      if (a.length < 1 / significance - 1) {
        logWarning(s"too few calibration samples (${a.length}) for significance $significance")
      }
    }
    super.predict(features, significance)
  }
  
  override def toString = {
    val algStr = alg.toString
    val alphStr = alphas
      .map(mpv => 
        "("+ mpv.map(_.toString).reduce(_+","+_)+")")
      .reduce(_+","+_)
    s"{$algStr},{$alphStr}"
    
  }
  
}

class AggregatedICPClassifier[A <: UnderlyingAlgorithm](
  val icps: Seq[ICPClassifierModel[A]])
  extends ICPClassifierModel[A] {

  override def mondrianPv(features: Vector) = {
    icps
      .flatMap { icp =>
        icp.mondrianPv(features)
          .zipWithIndex
      }
      .groupBy(_._2)
      .toArray
      .sortBy(_._1)
      .map {
        case (index, seq) =>
          val sortedSeq = seq.map(_._1).sorted
          val n = sortedSeq.length
          val median = if (n % 2 == 0) {
            (sortedSeq(n / 2 - 1) + sortedSeq(n / 2)) / 2
          } else {
            sortedSeq(n / 2)
          }
          median
      }
  }

}