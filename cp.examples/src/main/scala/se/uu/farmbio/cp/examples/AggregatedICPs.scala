package se.uu.farmbio.cp.examples

import scala.util.Random
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import se.uu.farmbio.cp.AggregatedICPClassifier
import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import se.uu.farmbio.cp.ICP
import scopt.OptionParser

object AggregatedICPs {

  case class Params(
    input: String = null,
    calibrationSize: Int = 0,
    numIterations: Int = 0,
    numOfICPs: Int = 0,
    master: String = null)

  def run(params: Params) {

    //Init Spark
    val conf = new SparkConf()
      .setAppName("AggregatedICPs")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)

    //Load and split data
    val Array(training, test) = MLUtils.loadLibSVMFile(sc, params.input)
      .randomSplit(Array(0.8, 0.2))

    //Train icps
    val icps = (1 to params.numOfICPs).map { _ =>
      val (calibration, properTraining) =
        ICP.splitCalibrationAndTraining(training, params.calibrationSize,
          bothClasses = true)
      //Train ICP
      val gbt = new GBT(properTraining.cache, params.numIterations)
      ICP.trainClassifier(gbt, numClasses = 2, calibration)
    }

    //Aggregate ICPs and perform tests
    val icp = new AggregatedICPClassifier(icps)
    val mondrianPvAndLabels = test.map { p =>
      (icp.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(mondrianPvAndLabels)
    println(metrics)

  }

  def main(args: Array[String]) {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("AggregatedICPs") {
      head("AggregatedICPs: an example of aggregated ICPs")
      opt[Int]("calibrationSize")
        .required()
        .text(s"size of calibration set (for each class)")
        .action((x, c) => c.copy(calibrationSize = x))
      opt[Int]("numIterations")
        .required()
        .text(s"number of GBT iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numOfICPs")
        .required()
        .text(s"number of ICPs to train")
        .action((x, c) => c.copy(numOfICPs = x))
      opt[String]("master")
        .text("spark master")
        .action((x, c) => c.copy(master = x))
      arg[String]("<input>")
        .required()
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input = x))

    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }

  }

}