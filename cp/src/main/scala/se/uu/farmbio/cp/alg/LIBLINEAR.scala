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

class LibLinAlg(
  private var training: Array[LabeledPoint],
  private val solverType: SolverType,
  private val regParam: Double,
  private val tol: Double)
  extends UnderlyingAlgorithm(null.asInstanceOf[RDD[LabeledPoint]]) {

  override def trainingProcedure(nullRdd: RDD[LabeledPoint]) = {
    val liblinModel = LIBLINEAR.train(training, solverType, regParam, tol)
    training = null
    val weights = liblinModel.getFeatureWeights
    val model = new SVMModel(Vectors.dense(weights).toSparse, 0.0)
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

}