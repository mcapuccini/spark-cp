package se.uu.farmbio.cp

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

private object BinaryClassificationICPMetrics {
  def allSignificances(
    mondrianPvAndLabels: RDD[(Array[Double], Double)]) = {
    mondrianPvAndLabels
      .flatMap(_._1)
      .filter(_ != 1).collect
  }
}

class BinaryClassificationICPMetrics private (
  private val mondrianPvAndLabels: RDD[(Array[Double], Double)],
  private val sigLowerBound: Double,
  private val sigUpperBound: Double,
  val sigBcast: Broadcast[Array[Double]]) extends Serializable {

  def this(
    mondrianPvAndLabels: RDD[(Array[Double], Double)],
    sigLowerBound: Double,
    sigUpperBound: Double,
    significances: Array[Double]) = {
    this(mondrianPvAndLabels,
      sigLowerBound,
      sigUpperBound,
      mondrianPvAndLabels
        .context
        .broadcast(significances))
  }

  def this(mondrianPvAndLabels: RDD[(Array[Double], Double)]) = {
    this(mondrianPvAndLabels,
      sigLowerBound = 0.0,
      sigUpperBound = 1.0,
      BinaryClassificationICPMetrics
        .allSignificances(mondrianPvAndLabels))
  }

  def this(
    mondrianPvAndLabels: RDD[(Array[Double], Double)],
    sigLowerBound: Double,
    sigUpperBound: Double) = {
    this(mondrianPvAndLabels,
      sigLowerBound,
      sigUpperBound,
      BinaryClassificationICPMetrics
        .allSignificances(mondrianPvAndLabels))
  }

  def this(
    mondrianPvAndLabels: RDD[(Array[Double], Double)],
    significances: Array[Double]) = {
    this(mondrianPvAndLabels,
      sigLowerBound = 0.0,
      sigUpperBound = 1.0,
      significances)
  }

  //Compute all of the metrics at once
  private val metricsBySig = mondrianPvAndLabels
    .flatMap {
      case (mPv, lab) =>
        //Apply filters to significances
        val filteredSigs = sigBcast.value
          .filter(_ != 1)
          .filter { pv =>
            sigLowerBound <= pv && pv <= sigUpperBound
          }
        //Perform text, and mark responses for each significance
        filteredSigs.map { sig =>
          val single = if (mPv.count(_ > sig) == 1) { 1L } else { 0L }
          val error = if (mPv(lab.toInt) <= sig) { 1L } else { 0L }
          val tp =
            if (single == 1L && error == 0L && lab == 1.0) { 1L }
            else { 0L }
          (sig, single, error, tp, lab)
        }
    }
    .groupBy(_._1)
    .map {
      case (sig, group) =>
        //Aggregate test responses
        val (_, totSing, totErr, totTp, _) = group.reduce((t1, t2) =>
          (sig, t1._2 + t2._2, t1._3 + t2._3, t1._4 + t2._4, t1._5))
        val n = group.size
        val efficiency = totSing.toDouble / n
        val errorRate = totErr.toDouble / n
        val recall = totTp.toDouble / group.count(_._5 == 1.0)
        val validity = errorRate <= sig
        (sig,
          efficiency,
          errorRate,
          recall,
          validity)
    }
    .collect
    .sortBy(_._1)

  val significances = metricsBySig.map {
    case (sig, eff, err, rec, vldt) => sig
  }

  def errorRateBySignificance = metricsBySig.map {
    case (sig, eff, err, rec, vldt) => (sig, err)
  }

  def efficiencyBySignificance = metricsBySig.map {
    case (sig, eff, err, rec, vldt) => (sig, eff)
  }

  def recallBySignificance = metricsBySig.map {
    case (sig, eff, err, rec, vldt) => (sig, rec)
  }
  
  def validityBySignificance = metricsBySig.map {
    case (sig, eff, err, rec, vldt) => (sig, vldt)
  }

  override def toString =
    "significances : " +
      this.significances
      .map(_.toString).reduce(_ + ", " + _) +
      "\nerror rate    : " +
      this.errorRateBySignificance
      .map(_._2.toString).reduce(_ + ", " + _) +
      "\nefficiency    : " +
      this.efficiencyBySignificance
      .map(_._2.toString)
      .reduce(_ + ", " + _) +
      "\nrecall        : " +
      this.recallBySignificance
      .map(_._2.toString)
      .reduce(_ + ", " + _)

}