package se.uu.farmbio.cp

import org.apache.spark.rdd.RDD

class BinaryClassificationICPMetrics(
  private val mondrianPvAndLabels: RDD[(Array[Double], Double)],
  private val sigLowerBound: Double,
  private val sigUpperBound: Double) extends Serializable {

  def this(mondrianPvAndLabels: RDD[(Array[Double], Double)]) = {
    this(mondrianPvAndLabels, 0.0, 1.0)
  }

  private lazy val effAndErrBySig = mondrianPvAndLabels
    .flatMap(_._1)
    .filter(_ != 1)
    .filter { pv =>
      sigLowerBound <= pv && pv <= sigUpperBound
    }
    .distinct
    .cartesian(mondrianPvAndLabels)
    .map {
      case (sig, (mPv, lab)) =>
        val single = if (mPv.count(_ > sig) == 1) { 1L } else { 0L }
        val error = if (mPv(lab.toInt) <= sig) { 1L } else { 0L }
        val tp =
          if (single == 1L && error == 0L && lab == 1.0) { 1L }
          else { 0L }
        (sig, single, error, tp, lab)
    }
    .groupBy(_._1)
    .map {
      case (sig, group) =>
        val (_, totSing, totErr, totTp, _) = group.reduce((t1, t2) =>
          (sig, t1._2 + t2._2, t1._3 + t2._3, t1._4 + t2._4, t1._5))
        val n = group.size
        val tpPlusFn = group.count(_._5 == 1.0)
        (sig, totSing.toDouble / n, totErr.toDouble / n,
          totTp.toDouble / tpPlusFn)
    }
    .collect //it will be as big as the calibration size
    .sortBy(_._1)

  def significances = effAndErrBySig.map {
    case (sig, eff, err, rec) => (sig)
  }

  def errorRateBySignificance = effAndErrBySig.map {
    case (sig, eff, err, rec) => (sig, err)
  }

  def efficiencyBySignificance = effAndErrBySig.map {
    case (sig, eff, err, rec) => (sig, eff)
  }

  def recallBySignificance = effAndErrBySig.map {
    case (sig, eff, err, rec) => (sig, rec)
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