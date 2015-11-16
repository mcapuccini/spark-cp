package se.uu.farmbio.cp

import scala.util.Random
import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.io.File
import org.apache.commons.io.FileUtils

private[cp] object ICPTest {

  def generate4ClassesData(instances: Int, seed: Long): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    Seq.fill(instances)((rnd.nextInt(100), rnd.nextInt(100))).map(r => {
      val label = if (r._1 < 50 && r._2 < 50) {
        0.0
      } else if (r._1 < 50) {
        1.0
      } else if (r._2 < 50) {
        2.0
      } else {
        3.0
      }
      new LabeledPoint(label, Vectors.dense(Array(r._1.toDouble, r._2.toDouble)))
    })
  }

  def generate4ClassesTrainCalibTest(significance: Double) = {
    val numClasses = 4
    val calibSamples = 4 * numClasses * (1 / significance - 1).ceil.toInt //4 times the minimum
    val training = ICPTest.generate4ClassesData(instances = 80,
      seed = Random.nextLong)
    val test = ICPTest.generate4ClassesData(instances = 20,
      seed = Random.nextLong)
    val calibration = ICPTest.generate4ClassesData(instances = calibSamples,
      seed = Random.nextLong)
      .toArray
    (training, calibration, test)
  }

  def generateBinaryData(instances: Int, seed: Long): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    Seq.fill(instances)(rnd.nextInt(100)).map(r => {
      val label = if (r < 50) {
        0.0
      } else {
        1.0
      }
      new LabeledPoint(label, Vectors.dense(r))
    })
  }

}

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
    private var model: Vector => Double,
    val training: Array[LabeledPoint])
  extends UnderlyingAlgorithm(model) {
  
  def this(input: RDD[LabeledPoint]) = {
    this(null.asInstanceOf[(Vector => Double)], input.collect)
    model = trainingProcedure(input)
  }

  override protected def trainingProcedure(input: RDD[LabeledPoint]) = {
    OneNNClassifier.createModel(training)
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

private[cp] class OneNNClassifierSerializer
  extends UnderlyingAlgorithmSerializer[OneNNClassifier] {

  override def serialize(oneNN: OneNNClassifier) = {
    oneNN.training.map { p =>
      s"${p.label}," +
        p.features.toArray.map(_.toString).reduce(_ + "," + _)
    }.reduce(_ + "\n" + _)
  }

  override def deserialize(oneNNstr: String) = {
    val training = oneNNstr.split("\\n").map { line =>
      val doubles = line.split(",").map(_.toDouble)
      val label = doubles(0)
      val features = Vectors.dense(doubles.takeRight(doubles.length - 1))
      new LabeledPoint(label, features)
    }
    val model = OneNNClassifier.createModel(training)
    new OneNNClassifier(model, training)
  }

}

@RunWith(classOf[JUnitRunner])
class ICPTest extends FunSuite with SharedSparkContext {

  Random.setSeed(11)

  test("ICP classification") {

    val significance = 0.20
    val errFracts = (0 to 100).map { _ =>
      val (training, calibration, test) = ICPTest.generate4ClassesTrainCalibTest(significance)
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
    val (calibration, trainingRDD) = ICP.splitCalibrationAndTraining(sc.parallelize(input), 30)
    val training = trainingRDD.collect
    val concat = calibration ++ training
    assert(calibration.length == 30)
    assert(training.length == 70)
    assert(concat.length == 100)
    concat.sortBy(_.label).zip(input).foreach {
      case (x, y) => assert(x == y)
    }

  }

  test("aggregated ICPs classification") {

    val significance = 0.20
    val test = ICPTest.generate4ClassesData(instances = 20,
      seed = Random.nextLong)
    val icps = (0 to 100).map { _ =>
      val (training, calibration, _) = ICPTest.generate4ClassesTrainCalibTest(significance)
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

  test("fair calibration and training split") {

    val input = (1 to 50).map(i => new LabeledPoint(0.0, Vectors.dense(i))) ++
      (51 to 100).map(i => new LabeledPoint(1.0, Vectors.dense(i)))
    val (calibration, trainingRDD) = ICP.splitCalibrationAndTraining(
      sc.parallelize(input), 15, bothClasses = true)
    val training = trainingRDD.collect
    val concat = calibration ++ training
    val count0 = calibration.count(_.label == 0.0)
    val count1 = calibration.count(_.label == 1.0)
    assert(count0 == 15)
    assert(count1 == 15)
    assert(calibration.length == 30)
    assert(training.length == 70)
    assert(concat.length == 100)
    concat.sortBy(_.features(0)).zip(input).foreach {
      case (x, y) => assert(x == y)
    }

  }

  test("binary classification metrics") {

    val Seq(training, calibration, test) =
      Seq(100, 10, 20).map { instances =>
        ICPTest.generateBinaryData(instances, Random.nextInt)
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
      (sig, efficiency, errorRate, recall)
    }

    val effBySig = effAndErrBySig.map(t => (t._1, t._2))
    assert(metrics.efficiencyBySignificance sameElements effBySig)
    val errRateBySig = effAndErrBySig.map(t => (t._1, t._3))
    assert(metrics.errorRateBySignificance sameElements errRateBySig)
    val recBySig = effAndErrBySig.map(t => (t._1, t._4))
    assert(metrics.recallBySignificance sameElements recBySig)

  }

  test("save/load model") {
    
    //Create tmpdir
    val tmpBase = FileUtils.getTempDirectory.getAbsolutePath
    val tmpDir = new File(s"$tmpBase/icptest${System.currentTimeMillis}")
    tmpDir.mkdir
    
    //Create some test data
    val significance = 0.2
    val (training, calibration, test) = ICPTest.generate4ClassesTrainCalibTest(significance)
    val alg = new OneNNClassifier(sc.parallelize(training))
    val model = ICP.trainClassifier(alg, numClasses = 4, calibration)
    
    //Save and load
    val serializer = new OneNNClassifierSerializer
    model.save(s"$tmpDir/model", serializer)
    val loadedModel = ICPClassifierModel.loadICPClassifierModel(s"$tmpDir/model", serializer)
    
    //Make sure they produce same predictions
    test.foreach { p =>
      assert(model.predict(p.features, significance) == loadedModel.predict(p.features, significance))
    }
    
    //Delete tmpdir
    FileUtils.deleteDirectory(tmpDir)
    
  }
  
  test("save/load aggregated model") {
    
    //Create tmpdir
    val tmpBase = FileUtils.getTempDirectory.getAbsolutePath
    val tmpDir = new File(s"$tmpBase/icptest${System.currentTimeMillis}")
    tmpDir.mkdir
    
    //Create some test data
    val significance = 0.20
    val test = ICPTest.generate4ClassesData(instances = 20,
      seed = Random.nextLong)
    val icps = (0 to 100).map { _ =>
      val (training, calibration, _) = ICPTest.generate4ClassesTrainCalibTest(significance)
      val alg = new OneNNClassifier(sc.parallelize(training))
      ICP.trainClassifier(alg, numClasses = 4, calibration)
    }
    val aggr = new AggregatedICPClassifier(icps)
    
    //Save and load
    val serializer = new OneNNClassifierSerializer
    aggr.save(s"$tmpDir/model", serializer)
    val loadedModel = ICPClassifierModel.loadAggregatedICPClassifier(s"$tmpDir/model", serializer)
    
    //Make sure they produce same predictions
    test.foreach { p =>
      assert(aggr.predict(p.features, significance) == loadedModel.predict(p.features, significance))
    }
    
    //Delete tmpdir
    FileUtils.deleteDirectory(tmpDir)
    
  }

}