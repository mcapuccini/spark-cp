package se.uu.farmbio.cp.alg

import org.apache.spark.SharedSparkContext
import org.scalatest.FunSuite
import se.uu.farmbio.cp.ICP
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import se.uu.farmbio.cp.ICPTest
import scala.util.Random

@RunWith(classOf[JUnitRunner])
class GBTTest extends FunSuite with SharedSparkContext {
  
  Random.setSeed(11)

  test("test performance") {
    val trainData = ICPTest.generateBinaryData(100, Random.nextLong)
    val testData = ICPTest.generateBinaryData(30, Random.nextLong)
    val (calibration, properTrain) = ICP.splitCalibrationAndTraining(sc.parallelize(trainData), 16)  
    val gbt = new GBT(properTrain, 30)
    val model = ICP.trainClassifier(gbt, numClasses=2, calibration)
    assert(TestUtils.testPerformance(model, sc.parallelize(testData)))
  }

}