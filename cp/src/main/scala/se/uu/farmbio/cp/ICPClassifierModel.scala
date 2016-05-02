package se.uu.farmbio.cp

import java.io.File
import java.io.IOException
import java.io.Serializable
import java.nio.file.Files
import java.nio.file.Paths

import scala.io.Source

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

object ICPClassifierModel {

  def loadICPClassifierModel[A <: UnderlyingAlgorithm](
    inputPath: String,
    serializer: UnderlyingAlgorithmSerializer[A]): ICPClassifierModel[A] = {
    val algStr = Source.fromFile(s"$inputPath/alg.txt").mkString
    val alg = serializer.deserialize(algStr)
    val alphasStr = Source.fromFile(s"$inputPath/alphas.txt").mkString
    val alphas = alphasStr.split("\\r?\\n")
      .map { a =>
        a.split(",").map(_.toDouble).toArray
      }.toSeq
    new ICPClassifierModelImpl[A](alg, alphas)
  }
  
  def loadAggregatedICPClassifier[A <: UnderlyingAlgorithm](
    inputPath: String,
    serializer: UnderlyingAlgorithmSerializer[A]) = {
    val inputFile = new File(inputPath)
    val icps = inputFile.listFiles.map { f =>
      loadICPClassifierModel(f.getAbsolutePath, serializer)
    }
    new AggregatedICPClassifier(icps)
  }

}

abstract class ICPClassifierModel[A <: UnderlyingAlgorithm](val alg: A)
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

  def save(outputPath: String, serializer: UnderlyingAlgorithmSerializer[A])

}

private[cp] class ICPClassifierModelImpl[A <: UnderlyingAlgorithm](
  override val alg: A,
  private val alphas: Seq[Array[Double]])
  extends ICPClassifierModel[A](alg) with Logging {

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

  override def save(
    outputPath: String,
    serializer: UnderlyingAlgorithmSerializer[A]) = {
    val dir = new File(outputPath)
    if (dir.mkdir) {
      val algStr = serializer.serialize(alg)
      val alphasStr = alphas.map { a =>
        a.map(_.toString).reduce(_ + "," + _)
      }.reduce(_ + "\n" + _)
      Files.write(Paths.get(s"${dir.getAbsolutePath}/alg.txt"),algStr.getBytes)
      Files.write(Paths.get(s"${dir.getAbsolutePath}/alphas.txt"),alphasStr.getBytes)
    } else {
      throw new IOException(s"Imposible to crate directory $outputPath")
    }
  }
  
}

class AggregatedICPClassifier[A <: UnderlyingAlgorithm](
  val icps: Seq[ICPClassifierModel[A]])
  extends ICPClassifierModel[A](icps.head.alg) {

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

  override def save(
    outputPath: String,
    serializer: UnderlyingAlgorithmSerializer[A]) = {
    val dir = new File(outputPath)
    if (dir.mkdir) {
      icps.zipWithIndex.foreach {
        case (icp, i) =>
          icp.save(dir.getAbsolutePath + s"/icp$i", serializer)
      }
    } else {
      throw new IOException(s"Imposible to crate directory $outputPath")
    }
  }

}