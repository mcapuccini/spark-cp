package se.uu.farmbio.cp.examples

import scopt.OptionParser
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import se.uu.farmbio.cp.liblinear.LIBLINEAR
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.Logging

object LibLinearTraining extends Logging {
  
  case class Params(
    trainInputPath: String = null,
    outputPath: String = null,
    calibrRatio: Double = 0.2,
    numberOfCPs: Int = 100,
    nofOutFiles: Int = 0,
    dfsBlockSize: String = "8M",
    master: String = null)
  
  def main(args: Array[String]) = {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("PubChemTraining") {
      head("LibLinearTraining: LIBINEAR training procedure")
      opt[Double]("calibrRatio")
        .text("fraction of calibration examples")
        .action((x, c) => c.copy(calibrRatio = x))
      opt[Int]("numberOfCPs")
        .text("number of CPs to train")
        .action((x, c) => c.copy(numberOfCPs = x))
      opt[String]("master")
        .text("spark master")
        .action((x, c) => c.copy(master = x))
      opt[Int]("nofOutFiles")
        .text("Number of output files. " + 
            "It can be equal to the parallelism level at most " + 
            "(defualt: as much as the parallelism level)")
        .action((x, c) => c.copy(nofOutFiles = x))
      opt[String]("dfsBlockSize")
        .text("It tunes the Hadoop dfs.block.size property (default:8M)")
        .action((x, c) => c.copy(dfsBlockSize = x))
      arg[String]("<input>")
        .required()
        .text("input path to training examples in LIBSVM format")
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
  
  def run(params: Params) {

    //Init Spark
    val conf = new SparkConf()
      .setAppName("LibLinearTraining")
    if (params.master != null) {
      conf.setMaster(params.master)
    }
    val sc = new SparkContext(conf)
    
    //Set and log dfs.block.size
    sc.hadoopConfiguration.set("dfs.block.size", params.dfsBlockSize)
    val blockSize = sc.hadoopConfiguration.get("dfs.block.size")
    logInfo(s"dfs.block.size = $blockSize")
    
    //Load data
    //This example assumes the training set to be relatively small
    //the model data generated will be big instead.
    val input = MLUtils.loadLibSVMFile(sc, params.trainInputPath)
    val trainingData = input.collect
    
    //Train the CPs
    val modelData = LIBLINEAR.trainAggregatedICPClassifier(
        sc, 
        trainingData, 
        params.calibrRatio, 
        params.numberOfCPs)
        
    //Save the model in a distributed fashion 
    modelData.save(params.outputPath, params.nofOutFiles)
    
    //Stop Spark
    sc.stop
    
  }

}