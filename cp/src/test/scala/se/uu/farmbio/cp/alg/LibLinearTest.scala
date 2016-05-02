package se.uu.farmbio.cp.alg

import scala.util.Random
import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.classification.SVMSuite
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.TestUtils
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class LibLinearTest extends FunSuite with SharedSparkContext {

  Random.setSeed(11)

  //Generate some test data
  val intercept = 0.0
  val weights = Array(1.0, 1.0)
  val training = SVMSuite.generateSVMInput(intercept, weights, 80, Random.nextInt)
  val calib = SVMSuite.generateSVMInput(intercept, weights, 20, Random.nextInt)
  val test = SVMSuite.generateSVMInput(intercept, weights, 30, Random.nextInt)

  test("test private utilities") {

    //Generate test
    val points = TestUtils.generateBinaryData(100, Random.nextInt).toArray

    //Calibration split
    val calibrationSizeP = 10
    val calibrationSizeN = 20
    val (proper, cali) = LIBLINEAR.calibrationSplit(points, calibrationSizeP, calibrationSizeN)
    assert(cali.filter(_.label == 1.0).size == calibrationSizeP)
    assert(cali.filter(_.label == 0.0).size == calibrationSizeN)
    assert((cali ++ proper).toSet sameElements points.toSet)

    //Balanced split
    val calibrationSize = 10
    val (properBal, caliBal) = LIBLINEAR.splitBalanced(points, calibrationSize)
    assert(caliBal.filter(_.label == 1.0).size == calibrationSize)
    assert(caliBal.filter(_.label == 0.0).size == calibrationSize)
    assert((caliBal ++ properBal).toSet sameElements points.toSet)

    //Fractional split
    val calibrationFraction = 0.2
    val (properFr, caliFr) = LIBLINEAR.splitFractional(points, calibrationFraction)
    assert(caliFr.filter(_.label == 1.0).size ==
      (points.filter(_.label == 1.0).size * calibrationFraction).toInt)
    assert(caliFr.filter(_.label == 0.0).size ==
      (points.filter(_.label == 0.0).size * calibrationFraction).toInt)
    assert((properFr ++ caliFr).toSet sameElements points.toSet)

  }

  test("test performance, ignore split, no aggregation") {
    
    //Train
    val alg = new LibLinAlg(training.toArray,
      SolverType.L2R_L2LOSS_SVC_DUAL,
      regParam = 1.0,
      tol = 0.01)
    val model = ICP.trainClassifier(alg, numClasses = 2, calib.toArray)
    
    assert(TestUtils.testPerformance(model, sc.parallelize(test)))
  
  }
  
  test("test performance, balanced split") {
    
    val model = LIBLINEAR.trainBinaryAggregatedClassifier(
        sc.parallelize(training), balancedCalibration=true, calibrationSize=16)
    assert(TestUtils.testPerformance(model, sc.parallelize(test)))
    
  }
  
  test("test performance, fractional split") {
    
    val model = LIBLINEAR.trainBinaryAggregatedClassifier(
        sc.parallelize(training), calibrationFraction=0.1)  
    assert(TestUtils.testPerformance(model, sc.parallelize(test)))
    
  }

}