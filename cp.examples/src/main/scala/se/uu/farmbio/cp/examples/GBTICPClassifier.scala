package se.uu.farmbio.cp.examples

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

import scopt.OptionParser
import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import se.uu.farmbio.cp.ICP

object GBTICPClassifier {

  case class Params(
    input: String = null,
    calibrationSize: Int = 0,
    numIterations: Int = 0,
    master: String = null)

  def run(params: Params) {

    //Init Spark
    val conf = new SparkConf()
      .setAppName("GBTICPClassifier")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)

    //Load and split data
    val Array(training, test) = MLUtils.loadLibSVMFile(sc, params.input)
      .randomSplit(Array(0.8, 0.2))
    val (calibration, properTraining) =
      ICP.splitCalibrationAndTraining(training, params.calibrationSize, bothClasses=true)

    //Train ICP
    val t0 = System.currentTimeMillis
    val gbt = new GBT(properTraining.cache, params.numIterations)
    val icp = ICP.trainClassifier(gbt, numClasses = 2, calibration)
    val t1 = System.currentTimeMillis
    
    //Compute and print metrics
    val mondrianPvAndLabels = test.map { p => 
      (icp.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(mondrianPvAndLabels)
    println(metrics)
    println(s"training took: ${t1-t0} millisec.")

    sc.stop

  }

  def main(args: Array[String]) {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("GBTICPClassifier") {
      head("GBTICPClassifier: an example of Gradient Boosted Trees ICP classification.")
      opt[Int]("calibrationSize")
        .required()
        .text(s"size of calibration set (for each class)")
        .action((x, c) => c.copy(calibrationSize = x))
      opt[Int]("numIterations")
        .required()
        .text(s"number of GBT iterations")
        .action((x, c) => c.copy(numIterations = x))
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