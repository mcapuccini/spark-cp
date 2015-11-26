package se.uu.farmbio.cp.alg

import scala.util.Random

import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import de.bwaldvogel.liblinear.Feature
import de.bwaldvogel.liblinear.FeatureNode
import de.bwaldvogel.liblinear.Linear
import de.bwaldvogel.liblinear.Parameter
import de.bwaldvogel.liblinear.Problem
import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.AggregatedICPClassifier
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.UnderlyingAlgorithm
import se.uu.farmbio.cp.UnderlyingAlgorithmSerializer

class LibLinAlg(
  val svmModel: SVMModel)
  extends UnderlyingAlgorithm(
    (features: Vector) => svmModel.predict(features)) {

  def this(
    training: Array[LabeledPoint],
    solverType: SolverType,
    regParam: Double,
    tol: Double) = {
    this(LIBLINEAR.train(training, solverType, regParam, tol))
  }

  override def nonConformityMeasure(newSample: LabeledPoint) = {
    val score = predictor(newSample.features)
    if (newSample.label == 1.0) {
      score
    } else {
      -score
    }
  }

}

object LibLinAlgSerializer extends UnderlyingAlgorithmSerializer[LibLinAlg] {
  override def serialize(alg: LibLinAlg): String = {
    alg.svmModel.intercept + "\n" +
      alg.svmModel.weights.toArray.map(_.toString).reduce(_ + "\n" + _)
  }
  override def deserialize(modelString: String): LibLinAlg = {
    val rowSplitted = modelString.split("\n").map(_.toDouble)
    val intercept = rowSplitted.head
    val weights = rowSplitted.tail
    val model = new SVMModel(Vectors.dense(weights), intercept)
    model.clearThreshold()
    new LibLinAlg(model)
  }
}

object LIBLINEAR {
  private[alg] def vectorToFeatures(v: Vector) = {
    val indices = v.toSparse.indices
    val values = v.toSparse.values
    indices
      .zip(values)
      .sortBy {
        case (i, v) => i
      }
      .map {
        case (i, v) => new FeatureNode(i + 1, v)
          .asInstanceOf[Feature]
      }
  }

  private[alg] def train(
    input: Array[LabeledPoint],
    solverType: SolverType,
    c: Double,
    tol: Double) = {

    //configure problem
    val problem = new Problem
    problem.l = input.length
    problem.n = input(0).features.size
    problem.x = input.map { p =>
      vectorToFeatures(p.features)
    }
    problem.y = input.map(_.label + 1.0)
    problem.bias = -1.0

    //train
    val parameter = new Parameter(solverType, c, tol)
    val libLinModel = Linear.train(problem, parameter)

    //convert to Spark SVMModel
    val weights = libLinModel.getFeatureWeights
    val svmModel = new SVMModel(Vectors.dense(weights).toSparse, 0.0)
    svmModel.clearThreshold
    svmModel

  }

  private[alg] def calibrationSplit(
    trainingData: Array[LabeledPoint],
    calibrationSizeP: Int,
    calibrationSizeN: Int) = {
    val shuffData = Random.shuffle(trainingData.toList)
    val positives = shuffData.filter { p => p.label == 1.0 }
    val negatives = shuffData.filter { p => p.label != 1.0 }
    val calibration = (
      positives.take(calibrationSizeP) ++
      negatives.take(calibrationSizeN))
      .toArray
    val properTraining = (
      positives.takeRight(positives.length - calibrationSizeP) ++
      negatives.takeRight(negatives.length - calibrationSizeN))
      .toArray
    (properTraining, calibration)
  }

  private[alg] def splitFractional(
    trainingData: Array[LabeledPoint],
    calibrationFraction: Double) = {
    val calibrationSizeP = (trainingData.count(_.label == 1.0) * calibrationFraction).toInt
    val calibrationSizeN = (trainingData.count(_.label != 1.0) * calibrationFraction).toInt
    calibrationSplit(trainingData, calibrationSizeP, calibrationSizeN)
  }

  private[alg] def splitBalanced(
    trainingData: Array[LabeledPoint],
    calibrationSize: Int) = {
    calibrationSplit(trainingData, calibrationSize, calibrationSize)
  }

  def trainBinaryAggregatedClassifier(
    trainingData: RDD[LabeledPoint],
    calibrationFraction: Double = 0.2,
    numberOfICPs: Int = 10,
    solverType: SolverType = SolverType.L2R_L2LOSS_SVC_DUAL,
    regParam: Double = 1,
    tol: Double = 0.01,
    balancedCalibration: Boolean = true,
    calibrationSize: Int = 16): AggregatedICPClassifier[LibLinAlg] = {

    //Broadcast the dataset
    val sc = trainingData.context
    val trainBroadcast = sc.broadcast(trainingData.collect)

    //Train ICPs for different calibration samples
    val icps = sc.parallelize((1 to numberOfICPs)).map { _ =>
      //Sample calibration
      val (properTraining, calibration) = if (balancedCalibration) {
        splitBalanced(trainBroadcast.value, calibrationSize)
      } else {
        splitFractional(trainBroadcast.value, calibrationFraction)
      }
      //Train ICP
      val alg = new LibLinAlg(
        properTraining,
        solverType,
        regParam,
        tol)
      ICP.trainClassifier(alg, numClasses = 2, calibration)
    }

    //Aggregate ICPs 
    new AggregatedICPClassifier(icps.collect)
  }

}
