package se.uu.farmbio.cp.liblinear

import scala.util.Random
import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.classification.SVMSuite
import org.apache.spark.mllib.regression.LabeledPoint
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import se.uu.farmbio.cp.ICPClassifierModel
import org.scalatest.junit.JUnitRunner
import java.io.File
import org.apache.commons.io.FileUtils

object LibLinTest {

  def testPerformance(
    model: ICPClassifierModel[LibLinAlg],
    test: Array[LabeledPoint],
    sig: Double = 0.2,
    minEff: Double = 0.7,
    minRec: Double = 0.7) = {
    val pvAndLab = test.map { p =>
      (model.mondrianPv(p.features), p.label)
    }
    val metrics = new BinaryClassificationICPMetrics(pvAndLab, Array(sig))
    val eff = metrics.efficiencyBySignificance(sig)
    val rec = metrics.recallBySignificance(sig)
    eff >= minEff && rec >= minRec
  }

}

@RunWith(classOf[JUnitRunner])
class LibLinTest extends FunSuite with SharedSparkContext {

  Random.setSeed(11)

  //Generate some test data
  val intercept = 0.0
  val weights = Array(3.0, -4.0)
  val training = SVMSuite.generateSVMInput(intercept, weights, 80, Random.nextInt)
    .toArray
  val calib = SVMSuite.generateSVMInput(intercept, weights, 20, Random.nextInt)
    .toArray
  val test = SVMSuite.generateSVMInput(intercept, weights, 30, Random.nextInt)
    .toArray

  test("test calibration split") {

    //Fractional split
    val calibrationFraction = 0.2
    val (properFr, caliFr) = LIBLINEAR.splitFractional(training, calibrationFraction)
    assert(caliFr.filter(_.label == 1.0).size ==
      (training.filter(_.label == 1.0).size * calibrationFraction).toInt)
    assert(caliFr.filter(_.label == 0.0).size ==
      (training.filter(_.label == 0.0).size * calibrationFraction).toInt)
    assert((properFr ++ caliFr).toSet sameElements training.toSet)

  }

  test("performance test") {
    //Train a model
    val model = LIBLINEAR.trainAggregatedICPClassifier(
      sc, training, numberOfICPs = 10)
    assert(LibLinTest.testPerformance(model, test))
  }

  test("binary classification metrics") {

    val model = LIBLINEAR.trainAggregatedICPClassifier(sc, training, numberOfICPs = 10)

    val mondrianPvAndLabels = test.map { p =>
      (model.mondrianPv(p.features), p.label)
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

  test("save/load") {

    //Train a model
    val model = LIBLINEAR.trainAggregatedICPClassifier(
      sc, training, numberOfICPs = 10)

    val tmp = System.getProperty("java.io.tmpdir")
    model.save(tmp + "/model")
    val loadedModel = AggregatedICPClassifier.load(tmp + "/model", sc)

    test.foreach { p =>
      assert(model.predict(p.features, 0.2) ==
        loadedModel.predict(p.features, 0.2))
    }
    
    FileUtils.deleteDirectory(new File(tmp + "/model"))

  }

}