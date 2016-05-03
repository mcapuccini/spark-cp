package se.uu.farmbio.cp

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import scala.util.Random
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.scalatest.mock.MockitoSugar
import org.mockito.Mockito._

private[cp] object OneNNClassifier {
  def createModel(training: Array[LabeledPoint]) = (features: Vector) => {
    val classAndDist = training.map(point =>
      (point.label, Vectors.sqdist(point.features, features)))
      .sorted
    classAndDist(0)._1 //return the class of the nearest neighbor
  }
}

//this will not work for big input RDDs, however this is just for testing purpose
private[cp] class OneNNClassifier(
    val model: Vector => Double,
    val training: Array[LabeledPoint])
  extends UnderlyingAlgorithm(model) {
  
  def this(input: RDD[LabeledPoint]) = {
    this(OneNNClassifier.createModel(input.collect), input.collect)
  }

  override def nonConformityMeasure(newSample: LabeledPoint) = {
    val filtTrain = training.filter(_.label == newSample.label)
    if (filtTrain.isEmpty) {
      //New class, then the new sample is very non conforming
      Double.MaxValue
    } else {
      //Avg distance from the previous samples of same class
      val distances = filtTrain.map(point =>
        Vectors.sqdist(point.features, newSample.features))
        .sorted
      distances.sum / filtTrain.length
    }
  }

}

@RunWith(classOf[JUnitRunner])
class ICPTest extends FunSuite with SharedSparkContext with MockitoSugar {

  Random.setSeed(11)

  test("ICP classification") {

    val significance = 0.20
    val errFracts = (0 to 100).map { _ =>
      val (training, calibration, test) = TestUtils.generate4ClassesTrainCalibTest(significance)
      val alg = new OneNNClassifier(sc.parallelize(training))
      val model = ICP.trainClassifier(alg, numClasses = 4, calibration)
      //compute error fraction
      val errors = test.count { p =>
        val region = model.predict(p.features, significance)
        !region.contains(p.label)
      }
      errors.toDouble / test.length.toDouble
    }

    val meanError = errFracts.sum / errFracts.length
    assert(meanError <= significance)

  }

  test("calibration and training split") {

    val input = (1 to 100).map(i => new LabeledPoint(i, Vectors.dense(i)))
    val (calibration, trainingRDD) = ICP.calibrationSplit(sc.parallelize(input), 30)
    val training = trainingRDD.collect
    val concat = calibration ++ training
    assert(calibration.length == 30)
    assert(training.length == 70)
    assert(concat.length == 100)
    concat.sortBy(_.label).zip(input).foreach {
      case (x, y) => assert(x == y)
    }

  }
  
  test("stratified calibration and training split") {
    
    val input = (1 to 1000).map(
        i => new LabeledPoint(
            if(i <= 200) 1.0 else 0.0, 
            Vectors.dense(i)))
    val (calibration, trainingRDD) = ICP.calibrationSplit(
        sc.parallelize(input), 300, stratified=true)
    val training = trainingRDD.collect
    val concat = calibration ++ training
    assert(calibration.filter(_.label==1.0).length == 60)
    assert(calibration.filter(_.label==0.0).length == 240)
    assert(training.length == 700)
    assert(concat.length == 1000)
    concat.sortBy(_.features.toArray(0)).zip(input).foreach {
      case (x, y) => assert(x == y)
    }
    
  }

  test("aggregated ICPs classification") {

    val significance = 0.20
    val test = TestUtils.generate4ClassesData(instances = 20,
      seed = Random.nextLong)
    val icps = (0 to 100).map { _ =>
      val (training, calibration, _) = TestUtils.generate4ClassesTrainCalibTest(significance)
      val alg = new OneNNClassifier(sc.parallelize(training))
      ICP.trainClassifier(alg, numClasses = 4, calibration)
    }
    val aggr = new AggregatedICPClassifier(icps)
    val errors = test.count { p =>
      val region = aggr.predict(p.features, significance)
      !region.contains(p.label)
    }
    val meanError = errors.toDouble / test.length.toDouble
    assert(meanError <= significance)

  }

  test("binary classification metrics") {

    val Seq(training, calibration, test) =
      Seq(100, 10, 20).map { instances =>
        TestUtils.generateBinaryData(instances, Random.nextInt)
      }

    val alg = new OneNNClassifier(sc.parallelize(training))
    val model = ICP.trainClassifier(alg, numClasses = 2, calibration.toArray)

    val mondrianPvAndLabels = sc.parallelize(test).map {
      p => (model.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(mondrianPvAndLabels)

    val effAndErrBySig = metrics.significances.map { sig =>
      val efficiency = test.count { p =>
        model.predict(p.features, sig).size == 1
      }.toDouble / test.length
      val errorRate = test.count { p =>
        !model.predict(p.features, sig).contains(p.label)
      }.toDouble / test.length
      val recall = test.count { p =>
        val set = model.predict(p.features, sig)
        set == Set(1.0) && p.label == 1.0
      }.toDouble / test.count(_.label == 1.0)
      val validity = errorRate <= sig
      (sig, efficiency, errorRate, recall, validity)
    }

    val effBySig = effAndErrBySig.map(t => (t._1, t._2))
    assert(metrics.efficiencyBySignificance sameElements effBySig)
    val errRateBySig = effAndErrBySig.map(t => (t._1, t._3))
    assert(metrics.errorRateBySignificance sameElements errRateBySig)
    val recBySig = effAndErrBySig.map(t => (t._1, t._4))
    assert(metrics.recallBySignificance sameElements recBySig)
    val valBySig = effAndErrBySig.map(t => (t._1, t._5))
    assert(metrics.validityBySignificance sameElements valBySig)

  }
  
  test("model to string") {
    
    val model = mock[OneNNClassifier]
    when(model.toString).thenReturn("1.0,2.0")
    val icp = new ICPClassifierModelImpl(model,Seq(
      Array(0.1,0.2,0.3),
      Array(0.3,0.2,0.1),
      Array(0.2,0.1,0.3)
    ))

    assert(icp.toString == 
      "[1.0,2.0],[(0.1,0.2,0.3),(0.3,0.2,0.1),(0.2,0.1,0.3)]")
    
  }

}