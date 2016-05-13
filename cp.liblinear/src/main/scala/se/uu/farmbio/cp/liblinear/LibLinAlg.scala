package se.uu.farmbio.cp.liblinear

import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import de.bwaldvogel.liblinear.Feature
import de.bwaldvogel.liblinear.FeatureNode
import de.bwaldvogel.liblinear.Linear
import de.bwaldvogel.liblinear.Parameter
import de.bwaldvogel.liblinear.Problem
import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.UnderlyingAlgorithm
import se.uu.farmbio.cp.Deserializer

object LibLinAlg {

  private def vectorToFeatures(v: Vector) = {
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

  private def train(
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
    val intercept = libLinModel.getBias
    val svmModel = new SVMModel(Vectors.dense(weights).toSparse, intercept)
    svmModel.clearThreshold
    svmModel

  }

}

object LibLinAlgDeserializer extends Deserializer[LibLinAlg] {
  override def deserialize(alg: String) = {
    val splitted = alg.split(",", 2)
    val intercept = splitted(0)
    val weights = splitted(1)
    val model = new SVMModel(Vectors.parse(weights).toSparse, intercept.toDouble)
    model.clearThreshold()
    new LibLinAlg(model)
  }  
}

class LibLinAlg(
  val svmModel: SVMModel)
  extends UnderlyingAlgorithm(
    (features: Vector) => svmModel.predict(features)) {

  def this(
    training: Array[LabeledPoint],
    solverType: SolverType,
    regParam: Double,
    tol: Double) = {
    this(LibLinAlg.train(training, solverType, regParam, tol))
  }

  override def nonConformityMeasure(newSample: LabeledPoint) = {
    val score = predictor(newSample.features)
    if (newSample.label == 1.0) {
      score
    } else {
      -score
    }
  }

  override def toString = {
    this.svmModel.intercept + "," +
      this.svmModel.weights.toString
  }

}