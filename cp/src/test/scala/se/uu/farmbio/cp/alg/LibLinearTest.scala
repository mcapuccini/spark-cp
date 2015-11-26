package se.uu.farmbio.cp.alg

import org.apache.spark.SharedSparkContext
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import se.uu.farmbio.cp.ICPTest
import scala.util.Random
import org.apache.commons.io.FileUtils
import se.uu.farmbio.cp.ICP
import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.ICPClassifierModel
import java.io.File
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import org.apache.spark.mllib.classification.SVMSuite

@RunWith(classOf[JUnitRunner])
class LibLinearTest extends FunSuite with SharedSparkContext {

  Random.setSeed(11)

  //Generate some test data
  val intercept=0.0 
  val weights=Array(3.0,2.0)
  val propTraining = SVMSuite.generateSVMInput(intercept, weights, 80, Random.nextInt)
  val calib = SVMSuite.generateSVMInput(intercept, weights, 20, Random.nextInt)
  val test = SVMSuite.generateSVMInput(intercept, weights, 30, Random.nextInt)

  //Train
  val alg = new LibLinAlg(propTraining.toArray,
    SolverType.L2R_L2LOSS_SVC_DUAL,
    regParam = 1.0,
    tol = 0.01)
  val model = ICP.trainClassifier(alg, numClasses = 2, calib.toArray)

  test("save/load LibLinAlg") {

    //Create tmpdir
    val tmpBase = FileUtils.getTempDirectory.getAbsolutePath
    val tmpDir = new File(s"$tmpBase/icptest${System.currentTimeMillis}")
    tmpDir.mkdir

    //Save to disk and load couple of times
    val serializer = LibLinAlgSerializer
    model.save(s"$tmpDir/firstModel", serializer)
    val model2 = ICPClassifierModel.loadICPClassifierModel(s"$tmpDir/firstModel", serializer)
    model2.save(s"$tmpDir/secondModel", LibLinAlgSerializer)
    val model3 = ICPClassifierModel.loadICPClassifierModel(s"$tmpDir/secondModel", serializer)

    //Perform test
    val significance = 0.2
    test.foreach { lp =>
      assert(
        model.predict(lp.features, significance).equals(
          model2.predict(lp.features, significance)) &&
          model.predict(lp.features, significance).equals(
            model3.predict(lp.features, significance)),
        "make sure that the non-saved and the loaded models make the same predictions")
    }

    //Delete tmpdir
    FileUtils.deleteDirectory(tmpDir)

  }

  test("test private utilities") {

    //Generate test
    val points = ICPTest.generateBinaryData(100, Random.nextInt).toArray

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

  test("test performance") {
    val sig = 0.2
    val pvAndLab = sc.parallelize(test.map { p =>
      (model.mondrianPv(p.features), p.label)
    })
    val metrics = new BinaryClassificationICPMetrics(pvAndLab, Array(sig))
    assert(metrics.efficiencyBySignificance(sig) >= 0.6)
    assert(metrics.recallBySignificance(sig) >= 0.6)
  }

}