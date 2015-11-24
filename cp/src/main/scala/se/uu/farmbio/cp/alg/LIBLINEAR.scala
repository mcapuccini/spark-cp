package se.uu.farmbio.cp.alg

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
import se.uu.farmbio.cp.UnderlyingAlgorithm
import se.uu.farmbio.cp.UnderlyingAlgorithmSerializer
import se.uu.farmbio.cp.AggregatedICPClassifier
import org.apache.spark.SparkContext
import scala.util.Random
import se.uu.farmbio.cp.ICP


class LibLinAlg(
  
  private var training: Array[LabeledPoint],
  private val solverType: SolverType,
  private val regParam: Double,
  private val tol: Double,
  private var model: SVMModel = null)
  extends UnderlyingAlgorithm(null.asInstanceOf[RDD[LabeledPoint]]) {
  
  def this(model: SVMModel){
    this(null, null, 0.0, 0.0, model);
  }
  
  //private var model: SVMModel = _
  def getModel = if(model!=null) model else throw new IllegalAccessException("Illegal access of model (not been created yet)");
  
  override def trainingProcedure(nullRdd: RDD[LabeledPoint]) = {
    val liblinModel = LIBLINEAR.train(training, solverType, regParam, tol)
    training = null
    val weights = liblinModel.getFeatureWeights
    model = new SVMModel(Vectors.dense(weights).toSparse, 0.0)
    model.clearThreshold
    model.predict
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



object LibLinAlgSerializer extends UnderlyingAlgorithmSerializer[LibLinAlg]{
  override def serialize(alg: LibLinAlg): String = {
    alg.getModel.intercept + "\n" +
    alg.getModel.weights.toArray.map(_.toString).reduce(_+"\n"+_)
  }
  override def deserialize(modelString: String): LibLinAlg ={
    val rowSplitted= modelString.split("\n").map(_.toDouble);
    val intercept = rowSplitted.head;
    val weights = rowSplitted.tail;
    val model = new SVMModel(Vectors.dense(weights), intercept);
    new LibLinAlg(model);
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
  
  case class LibLinParams(
    calibrationSize: Int = 16,
    nICPs: Int = 30,
    solverType: SolverType = SolverType.L2R_L2LOSS_SVC_DUAL,
    regParam: Double = 1,
    tol: Double = 0.01)
  
  def trainAggregatedClassifier(params: LibLinParams, training_data: RDD[LabeledPoint], sc: SparkContext): AggregatedICPClassifier[LibLinAlg] = {

    //Train icps
    val trainBroadcast = sc.broadcast(training_data.collect)
    val icps = sc.parallelize((1 to params.nICPs)).map { _ =>
      //Sample calibration
      val shuffData = Random.shuffle(trainBroadcast.value.toList)
      val positives = shuffData.filter { p => p.label == 1.0 }
      val negatives = shuffData.filter { p => p.label != 1.0 }
      val calibration = (
        positives.take(params.calibrationSize) ++
        negatives.take(params.calibrationSize))
        .toArray
      val properTraining = (
        negatives.takeRight(negatives.length - params.calibrationSize) ++
        positives.takeRight(positives.length - params.calibrationSize))
        .toArray
      //Train ICP
      val alg = new LibLinAlg(
        properTraining,
        params.solverType,
        params.regParam,
        params.tol)
      ICP.trainClassifier(alg, numClasses = 2, calibration)
    }

    //Aggregate ICPs 
    return new AggregatedICPClassifier(icps.collect)
  }
  
  /**
   * takeFractions will take a List of any type and an Array with fractions wanted in
   * each output-list. The fractions can be given in any "format", if 3 equally big lists
   * are wanted you can specify fractions as Array(1.0,1.0,1.0), Array(100, 100, 100) or similar.
   * When the list is not perfectly divisible by the fractions, remaining items will be added (one at a time)
   * to the fractions in order of the fraction-size. 
   * @param list			The list to be splitted into sub-lists
   * @param fractions	The required fractions, no more than 10 allowed 
   * @return 					An Array with sub-lists from the given list, the list will be randomized
   */
  def takeFractions[T](list: List[T], fractions: Array[Double]): Array[List[T]] ={
    if(fractions.length <= 1){
      return Array(Random.shuffle(list));
    }
    if(fractions.length > 10){
      throw new IllegalArgumentException("number of fractions are not allowed to be greater than 10");
    }
    else if(fractions.min <0){
      throw new IllegalArgumentException("The fractions should all be equal or greater than 0");
    }
    else if(fractions.sum == 0){
      throw new IllegalArgumentException("The sum of the fractions should be greater than 0"); 
    }
    
    val sumFractions = fractions.sum;
    val numRecords = list.size;
    var shuffData = Random.shuffle(list);
    
    var lists = fractions.map { frac => 
      val (currFraction, new_shuffData) = shuffData.splitAt((frac*numRecords/sumFractions).toInt); 
      shuffData=new_shuffData; 
      currFraction 
    }
    
    // for the data left due to list not being perfectly divisible by the fractions
    var fracs = fractions;
    for (data <- shuffData){
      val biggestIndex = fracs.indexOf(fracs.max);
      fracs(biggestIndex) = fracs(biggestIndex)-100;
      lists(biggestIndex) = data::lists(biggestIndex)
    }

    lists
  }

}
