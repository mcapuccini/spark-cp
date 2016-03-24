package se.uu.farmbio.cp.examples

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.alg.SVM
import se.uu.farmbio.cp.ICPClassifierModel
import se.uu.farmbio.cp.AggregatedICPClassifier
import se.uu.farmbio.cp.alg.SVMSerializer
import scopt.OptionParser

object PubChemTraining {

  case class Params(
    trainInputPath: String = null,
    outputPath: String = null,
    clSize: Int = 100,
    nCPs: Int = 100,
    master: String = null)

  def run(params: Params) {

    //Init Spark
    val conf = new SparkConf()
      .setAppName("PubChemTraining")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)

    //Load file
    val pubchem = MLUtils.loadLibSVMFile(sc, params.trainInputPath)

    //Train with SVM
    val cps = (1 to params.nCPs).map { _ =>
      //Sample calibration
      val (calibration, training) = ICP.calibrationSplit(
        pubchem,
        params.clSize,
        stratified = true)
      ICP.trainClassifier(
        new SVM(training,regParam=0.0),
        numClasses = 2,
        calibration)
    }

    //Save as aggregated ICP
    new AggregatedICPClassifier(cps)
      .save(params.outputPath, SVMSerializer)

  }

  def main(args: Array[String]) = {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("PubChemTraining") {
      head("PubChemTraining: training procedure for unbalanced datasets, e.g. PubChem tox.")
      opt[Int]("clSize")
        .text("size of calibration set")
        .action((x, c) => c.copy(clSize = x))
      opt[Int]("nCPs")
        .text("number of CPs to train")
        .action((x, c) => c.copy(nCPs = x))
      opt[String]("master")
        .text("spark master")
        .action((x, c) => c.copy(master = x))
      arg[String]("<input>")
        .required()
        .text("input path to training labeled examples in LIBSVM format")
        .action((x, c) => c.copy(trainInputPath = x))
      arg[String]("<output>")
        .required()
        .text("output path to save CPs")
        .action((x, c) => c.copy(outputPath = x))

    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }

  }

}