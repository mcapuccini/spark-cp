package se.uu.farmbio.cp.alg

import scala.util.Random
import org.apache.spark.SharedSparkContext
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.TestUtils
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class GBTTest extends FunSuite with SharedSparkContext {
  
  Random.setSeed(11)

  test("test performance") {
    val trainData = TestUtils.generateBinaryData(100, 11)
    val testData = TestUtils.generateBinaryData(30, 22)
    val (calibration, properTrain) = ICP.calibrationSplit(sc.parallelize(trainData), 16)  
    val gbt = new GBT(properTrain, 30)
    val model = ICP.trainClassifier(gbt, numClasses=2, calibration)
    assert(TestUtils.testPerformance(model, sc.parallelize(testData)))
  }

}