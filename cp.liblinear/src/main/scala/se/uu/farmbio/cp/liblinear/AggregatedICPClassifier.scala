package se.uu.farmbio.cp.liblinear

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import se.uu.farmbio.cp.ICPClassifierModel
import org.apache.commons.lang.NotImplementedException
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext

object AggregatedICPClassifier {

  def load(path: String, sc: SparkContext) = {
    val icps = sc.textFile(path)
      .map(ICPClassifierModel.deserialize(_, LibLinAlgDeserializer))
    new AggregatedICPClassifier(icps)
  }

}

class AggregatedICPClassifier(
  private val icps: RDD[ICPClassifierModel[LibLinAlg]])
  extends ICPClassifierModel[LibLinAlg] {

  val cachedICPs = icps.cache

  override def mondrianPv(features: Vector) = {
    cachedICPs
      .flatMap { icp =>
        icp.mondrianPv(features)
          .zipWithIndex
      }
      .collect //we expect to aggregate up to 100 ICPs
      .groupBy(_._2)
      .toArray
      .sortBy(_._1)
      .map {
        case (index, seq) =>
          val sortedSeq = seq.map(_._1).toArray.sorted
          val n = sortedSeq.length
          val median = if (n % 2 == 0) {
            (sortedSeq(n / 2 - 1) + sortedSeq(n / 2)) / 2
          } else {
            sortedSeq(n / 2)
          }
          median
      }
  }

  def save(path: String, coalesce: Int = 0) = {
    var serialICPs = cachedICPs.map(_.toString)
    if (coalesce > 0) {
      serialICPs = serialICPs.coalesce(coalesce)
    }
    serialICPs.saveAsTextFile(path)
  }

}