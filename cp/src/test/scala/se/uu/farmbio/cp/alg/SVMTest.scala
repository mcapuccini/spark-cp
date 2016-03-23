package se.uu.farmbio.cp.alg

import scala.util.Random
import org.apache.spark.SharedSparkContext
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.TestUtils
import org.scalatest.junit.JUnitRunner
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vectors

@RunWith(classOf[JUnitRunner])
class SVMTest extends FunSuite with SharedSparkContext {

  test("test performance") {
    val trainData = TestUtils.generateBinaryData(100, 11)
    val testData = TestUtils.generateBinaryData(30, 22)
    val (calibration, properTrain) = ICP.calibrationSplit(sc.parallelize(trainData), 16)  
    val svm = new SVM(properTrain, 30)
    val model = ICP.trainClassifier(svm, numClasses=2, calibration)
    assert(TestUtils.testPerformance(model, sc.parallelize(testData)))
  }
  
  test("serialize/deserialize") {
    
    val svmModel = new SVMModel(Vectors.dense(Array(0.1,0.2,0.3)),0.0)
    val alg = new SVM(svmModel)
    val algSerial = SVMSerializer.serialize(alg)
    val toTest = SVMSerializer.deserialize(algSerial)
    assert(toTest.model.weights == svmModel.weights)
    assert(toTest.model.intercept == svmModel.intercept)
    
  }

}