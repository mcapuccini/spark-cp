package se.uu.farmbio.cp.examples

import scopt.OptionParser
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object Splitter {

  case class Params(
    input: String = null,
    ratio: Double = 0.0,
    master: String = null)
    
  def run(params: Params) {
    
    //Init Spark
    val conf = new SparkConf()
      .setAppName("Splitter")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)
    
    //Load and split data
    val Array(training, test) = MLUtils.loadLibSVMFile(sc, params.input)
      .randomSplit(Array(params.ratio, 1.0-params.ratio))
      
    MLUtils.saveAsLibSVMFile(training, params.input+s".${params.ratio}.svm") 
    val roundRatio = BigDecimal(1.0-params.ratio)
      .setScale(1, BigDecimal.RoundingMode.HALF_UP)
    MLUtils.saveAsLibSVMFile(test, params.input+s".$roundRatio.svm") 
      
    sc.stop
    
  }
    
  def main(args: Array[String]) {
    
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Splitter") {
      head("Splitter: randomly split data into training and test.")
      opt[Double]("ratio")
        .required()
        .text("split ratio")
        .action((x, c) => c.copy(ratio = x))
      opt[String]("master")
        .text("spark master")
        .action((x, c) => c.copy(master = x))
      arg[String]("<input>")
        .required()
        .text("input path to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
    
  }

}