package se.uu.farmbio.cp.examples

import org.apache.spark.SparkConf
import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import org.apache.spark.SparkContext
import se.uu.farmbio.cp.ICP
import org.apache.spark.mllib.util.MLUtils
import scopt.OptionParser
import se.uu.farmbio.cp.alg.GBT

object ICPRunningTime {

  case class Params(
    input: String = null,
    calibrationSize: Int = 0,
    numIterations: Int = 0,
    inputFraction: Double = 0.0,
    master: String = null)

  def run(params: Params) {

    //Init Spark
    val conf = new SparkConf()
      .setAppName("ICPRunningTime")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)

    //Load and split data
    val training = MLUtils.loadLibSVMFile(sc, params.input)
      .sample(withReplacement = false, fraction = params.inputFraction)
    val (calibration, properTraining) =
      ICP.calibrationSplit(training, params.calibrationSize)

    //Train ICP
    val t0 = System.currentTimeMillis
    val gbt = new GBT(properTraining.cache, params.numIterations)
    val icp = ICP.trainClassifier(gbt, numClasses = 2, calibration)
    val t1 = System.currentTimeMillis
    
    println(s"training took: ${t1 - t0} millisec.")

    sc.stop

  }

  def main(args: Array[String]) {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("ICPRunningTime") {
      head("ICPRunningTime: it measures ICP with GBT training time for a fraction of the input.")
      opt[Int]("calibrationSize")
        .required()
        .text(s"size of calibration set (for each class)")
        .action((x, c) => c.copy(calibrationSize = x))
      opt[Int]("numIterations")
        .required()
        .text(s"number of GBT iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("inputFraction")
        .required()
        .text(s"input fraction to use for measuring training time")
        .action((x, c) => c.copy(inputFraction = x))  
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