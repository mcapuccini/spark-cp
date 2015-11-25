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
  private var svmModel: SVMModel,
  private var predModel: (Vector => Double))
  extends UnderlyingAlgorithm(predModel) { //null.asInstanceOf[RDD[LabeledPoint]]) {

  private var solverType: SolverType = _
  private var regParam: Double = _
  private var tol: Double = _
  private var training: Array[LabeledPoint] = _

  def this(training: Array[LabeledPoint],
           solverType: SolverType,
           regParam: Double,
           tol: Double) {
    this(null, null) //.asInstanceOf[SVMModel]
    this.solverType = solverType
    this.regParam = regParam
    this.tol = tol
    this.training = training
    model = trainingProcedure(null.asInstanceOf[RDD[LabeledPoint]])
  }

  //getter for the svm-model (used for serialization)
  def getSVMModel = if (svmModel != null) {
    svmModel
  } else {
    throw new IllegalAccessException("Illegal access of model (not been created yet)")
  }

  override def trainingProcedure(nullRdd: RDD[LabeledPoint]) = {
    val liblinModel = LIBLINEAR.train(training, solverType, regParam, tol)
    training = null // No need to store the training-examples any more
    val weights = liblinModel.getFeatureWeights
    svmModel = new SVMModel(Vectors.dense(weights).toSparse, 0.0)
    svmModel.clearThreshold
    model = svmModel.predict
    model
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
    alg.getSVMModel.intercept + "\n" +
      alg.getSVMModel.weights.toArray.map(_.toString).reduce(_ + "\n" + _)
  }
  override def deserialize(modelString: String): LibLinAlg = {
    val rowSplitted = modelString.split("\n").map(_.toDouble)
    val intercept = rowSplitted.head
    val weights = rowSplitted.tail
    val model = new SVMModel(Vectors.dense(weights), intercept)
    model.clearThreshold()
    new LibLinAlg(model, model.predict)
  }
}

object LIBLINEAR {
  def vectorToFeatures(v: Vector) = {
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

  def train(input: Array[LabeledPoint],
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

    val parameter = new Parameter(solverType, c, tol)
    Linear.train(problem, parameter)

  }
  
  def trainAggregatedClassifier(
    trainingData: RDD[LabeledPoint],
    fractionalCalibration: Boolean = true,
    calibrationSize: Int = 16,
    calibrationFraction: Double = 0.2,
    numberOfICPs: Int = 10,
    solverType: SolverType = SolverType.L2R_L2LOSS_SVC_DUAL,
    regParam: Double = 1,
    tol: Double = 0.01): AggregatedICPClassifier[LibLinAlg] = {

    val sc = trainingData.context
    //Train icps
    val trainBroadcast = sc.broadcast(trainingData.collect)
    val icps = sc.parallelize((1 to numberOfICPs)).map { _ =>
      //Sample calibration
      val (calibration, properTraining) = if (fractionalCalibration) {
        takeFractionBinaryStratisfied(trainBroadcast.value, calibrationFraction)
      } else {
        takeAbsoluteBinaryStratisfied(trainBroadcast.value, calibrationSize)
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
    return new AggregatedICPClassifier(icps.collect)
  }

  /**
   * takeAbsolute takes absolute numbers of elements from
   * a list and return all remaining elements
   * as a last index in the output Array. The given
   * output array with lists will have one more index
   * compared to the numToTake-input array.
   * @param list
   */
  private[alg] def takeAbsolute[T](
    list: List[T],
    numToTake: Array[Int]): Array[List[T]] = {
    if (numToTake.sum > list.length) {
      throw new IllegalArgumentException(
        "specified number to take is too large compared to the size of the list")
    }
    val listSize = list.length
    var shuffData = Random.shuffle(list)
    var lists: Array[List[T]] = numToTake.map { n =>
      val data = shuffData.take(n)
      shuffData = shuffData.takeRight(shuffData.length - n)
      data
    }
    lists = lists ++ Array(shuffData)
    lists

  }

  /**
   * takeFractions will take a List of any type and an Array with fractions wanted in
   * each output-list. The fractions can be given in any "format", if 3 equally big lists
   * are wanted you can specify fractions as Array(1.0,1.0,1.0), Array(100, 100, 100) or similar.
   * When the list is not perfectly divisible by the fractions,
   * remaining items will be added (one at a time)
   * to the fractions in order of the fraction-size.
   * @param list			The list to be splitted into sub-lists
   * @param fractions	The required fractions, no more than 10 allowed
   * @return 					An Array with sub-lists from the given list, the list will be randomized
   */
  private[alg] def takeFractions[T](
    list: List[T],
    fractions: Array[Double]): Array[List[T]] = {
    if (fractions.length <= 1) {
      return Array(Random.shuffle(list))
    }
    if (fractions.length > 10) {
      throw new IllegalArgumentException(
        "number of fractions are not allowed to be greater than 10")
    }
    if (fractions.min < 0) {
      throw new IllegalArgumentException(
        "The fractions should all be equal or greater than 0")
    }
    if (fractions.sum == 0) {
      throw new IllegalArgumentException(
        "The sum of the fractions should be greater than 0")
    }

    val sumFractions = fractions.sum
    val numRecords = list.size
    var shuffData = Random.shuffle(list)

    var lists = fractions.map { frac =>
      val (currFraction, newShuffData) =
        shuffData.splitAt((frac * numRecords / sumFractions).toInt)
      shuffData = newShuffData
      currFraction
    }

    // for the data left due to list not being perfectly divisible by the fractions
    var fracs = fractions
    for (data <- shuffData) {
      val biggestIndex = fracs.indexOf(fracs.max)
      fracs(biggestIndex) = fracs(biggestIndex) - 100
      lists(biggestIndex) = data :: lists(biggestIndex)
    }

    lists
  }

  /**
   * takeFractionBinaryStratisfied splits your data based
   * on the two outputs of your data (label == 1.0 or label != 1.0),
   * the splits will be made according to the calibrationFraction
   * specified and in stratisfied fashion.
   */
  def takeFractionBinaryStratisfied(
    data: Array[LabeledPoint],
    calibrationFraction: Double): (Array[LabeledPoint], Array[LabeledPoint]) = {
    if (calibrationFraction > 1 || calibrationFraction < 0) {
      throw new IllegalArgumentException("The calibrationFraction must be between 0 and 1")
    }

    val positives = data.filter { p => p.label == 1.0 }.toList
    val negatives = data.filter { p => p.label != 1.0 }.toList
    val Array(posCalib, posProper) =
      takeFractions(positives, Array(calibrationFraction, 1 - calibrationFraction))
    val Array(negCalib, negProper) =
      takeFractions(negatives, Array(calibrationFraction, 1 - calibrationFraction))

    ((posCalib ++ negCalib).toArray, (posProper ++ negProper).toArray)
  }

  /**
   * takeFractionBinaryStratisfied splits your data based
   * on the two outputs of your data (label == 1.0 or label != 1.0),
   * the splits will be made according to the calibrationFraction
   * specified and in stratisfied fashion.
   */
  def takeAbsoluteBinaryStratisfied(
    data: Array[LabeledPoint],
    calibrationSize: Int): (Array[LabeledPoint], Array[LabeledPoint]) = {
    if (calibrationSize > data.length) {
      throw new IllegalArgumentException(
        "The calibrationSize is set to be larger than the data-size")
    }
    val positives = data.filter { p => p.label == 1.0 }.toList
    val negatives = data.filter { p => p.label != 1.0 }.toList
    val Array(posCalib, posProper) = takeAbsolute(positives, Array(calibrationSize))
    val Array(negCalib, negProper) = takeAbsolute(negatives, Array(calibrationSize))

    ((posCalib ++ negCalib).toArray, (posProper ++ negProper).toArray)
  }

}
