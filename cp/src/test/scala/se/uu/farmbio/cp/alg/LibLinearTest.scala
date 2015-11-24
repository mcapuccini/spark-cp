package se.uu.farmbio.cp.alg

import java.io.File
import org.apache.commons.io.FileUtils
import org.apache.spark.SharedSparkContext
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import de.bwaldvogel.liblinear.SolverType
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.ICPClassifierModel
import se.uu.farmbio.cp.ICPTest
import se.uu.farmbio.cp.alg.LIBLINEAR.LibLinParams
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class LibLinearTest extends FunSuite with SharedSparkContext {

  val smallTestData = ICPTest.generateBinaryData(10, 1L).toArray
  val largeTestData = ICPTest.generateBinaryData(100, 1L).toArray
  val largeTestData_num = largeTestData.length
  val largeTestData_NumPos = largeTestData.filter { p => p.label == 1.0 }.length
  val largeTestData_NumNeg = largeTestData.filter { p => p.label != 1.0 }.length

  test("util function takeFractions") {
    val origList = (1 to 50).toList

    // equal partitions:
    val outputLists = LIBLINEAR.takeFractions(origList, Array(100, 100, 100, 100, 100))
    assert(outputLists.length == 5,
      "The number of created sub-lists should correspond to the given number")
    outputLists.foreach { subList =>
      assert(subList.length == 10, "Each sub-list should be correct length")
    }
    assert(origList.intersect(outputLists.flatten.toSeq).length == origList.length,
      "The same numbers should be in the outputList as in the original list")

    // un-equal partitions:
    val unequalLists = LIBLINEAR.takeFractions(origList, Array(5, 15, 25, 5))
    assert(unequalLists.length == 4)
    assert(unequalLists(0).size == 5)
    assert(unequalLists(1).size == 15)
    assert(unequalLists(2).size == 25)
    assert(unequalLists(3).size == 5)

    // when fractions is not perfectly divisible 
    val dontAddUp = LIBLINEAR.takeFractions(origList, Array(100, 100, 100))
    assert(dontAddUp.length == 3)
    assert(dontAddUp.flatten.length == origList.length,
      "When fractions are not perfectly dividing the input-list, should not be a problem")

    //corner-case: 0 or 1 fraction specified 
    val smallList = (1 to 5).toList
    val outSmall = LIBLINEAR.takeFractions(smallList, Array())
    assert(!outSmall(0).equals(smallList),
      "In cornercase (0 or 1 fractions given), the output should be randomized")
    val output0Fracs = LIBLINEAR.takeFractions(smallList, Array())

    assert(output0Fracs.length == 1,
      "0 fracs should just return a randomized list")
    assert(output0Fracs(0).length == smallList.size)
    assert(LIBLINEAR.takeFractions(smallList, Array(100)).length == 1,
      "")

    //corner-case: more than 10 fractions specified (should throw exception)
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeFractions(List(), (1 to 11).toList.map(_.toDouble).toArray)
    }

    // fractions should all be positive
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeFractions(List(), Array(-1, 0, 50, 1))
    }
    // sum of all fractions must be greater than 0
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeFractions(List(), Array(0, 0))
    }
  }

  test("util function takeAbsolute") {
    val lists = LIBLINEAR.takeAbsolute((1 to 5).toList, Array(1, 1))
    assert(lists.length == 3,
      "The number of output lists should be correct" +
        "(1 more than the specified numbers to take, " +
        "last index containing all remaining elements)")
    assert(lists.flatten.intersect((1 to 5).toList).length == 5,
      "The output values should be the once given as input")
    assert(lists(0).length == 1)
    assert(lists(1).length == 1)
    assert(lists(2).length == 3, "The remaning elements should be in the last index")

    // do another one, which is "corner-case"
    val lists2 = LIBLINEAR.takeAbsolute((1 to 5).toList, Array(0, 0))
    assert(lists2.length == 3,
      "The number of output lists should be correct" +
        "(1 more than the specified numbers to take, " +
        "last index containing all remaining elements)")
    assert(lists2.flatten.intersect((1 to 5).toList).length == 5,
      "The output values should be the once given as input")
    assert(lists2(0).length == 0)
    assert(lists2(1).length == 0)
    assert(lists2(2).length == 5, "The remaning elements should be in the last index")

    // non valid arguments
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeAbsolute(List(), Array(1))
    }
  }

  test("trainAggregatedClassifier") {
    val params = new LibLinParams()
    val (test, training) = LIBLINEAR.takeFractionBinaryStratisfied(largeTestData, 0.2)
    val rddTraining = sc.parallelize(training)
    val aggr = LIBLINEAR.trainAggregatedClassifier(params, rddTraining)

    test.foreach { lp =>
      aggr.predict(lp.features, significance = 0.3)
    }

    test.foreach { lp =>
      aggr.mondrianPv(lp.features)
    }
  }

  test("save/load LibLinAlg") {
    //Create tmpdir
    val tmpBase = FileUtils.getTempDirectory.getAbsolutePath
    val tmpDir = new File(s"$tmpBase/icptest${System.currentTimeMillis}")
    tmpDir.mkdir

    val params = new LibLinParams()
    val Array(test, calib, propTraining) =
      LIBLINEAR.takeFractions(largeTestData.toList, Array(0.2, 0.2, .6))
    val alg = new LibLinAlg(propTraining.toArray,
      SolverType.L2R_L2LOSS_SVC_DUAL,
      1,
      0.01)
    val model = ICP.trainClassifier(alg, numClasses = 2, calib.toArray)

    //val rddTraining = sc.parallelize(training)
    //val aggr = LIBLINEAR.trainAggregatedClassifier(params, rddTraining)

    //save:
    val serializer = LibLinAlgSerializer
    model.save(s"$tmpDir/firstModel", serializer)
    val model2 = ICPClassifierModel.loadICPClassifierModel(s"$tmpDir/firstModel", serializer)

    //save again:
    model2.save(s"$tmpDir/secondModel", LibLinAlgSerializer)
    val model3 = ICPClassifierModel.loadICPClassifierModel(s"$tmpDir/secondModel", serializer)

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

  test("takeFractionBinaryStratisfied") {

    val frac = 0.2
    val (calib, propTrain) = LIBLINEAR.takeFractionBinaryStratisfied(largeTestData, frac)
    val posCalib = calib.filter { lp => lp.label == 1.0 }.length
    val negCalib = calib.filter { lp => lp.label != 1.0 }.length

    assert(math.abs(posCalib - frac * largeTestData_NumPos) < 2,
      "The number of positive calibration datapoints should be correct")

    assert(math.abs(negCalib - frac * largeTestData_NumNeg) < 2,
      "The number of negative calibration datapoints should be correct")

    assert(calib.length + propTrain.length == largeTestData_num,
      "The total number of records should not be changed")

    // if the calibrationFraction is set to be larger than 1
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeFractionBinaryStratisfied(smallTestData, 1.1)
    }
    // if the calibrationFraction is set to be smaller than the 0
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeFractionBinaryStratisfied(smallTestData, -1)
    }

  }

  test("takeAbsoluteBinaryStratisfied") {
    val numCalib = 10
    val (calib, propTrain) = LIBLINEAR.takeAbsoluteBinaryStratisfied(largeTestData, numCalib)
    val posCalib = calib.filter { lp => lp.label == 1.0 }.length
    val negCalib = calib.filter { lp => lp.label != 1.0 }.length

    assert(posCalib == numCalib,
      "The number of positive calibration samples should be correct")
    assert(negCalib == numCalib,
      "The number of negative calibration samples should be correct")
    assert(calib.length + propTrain.length == largeTestData_num,
      "The total number of records should not be changed")

    // if the calibrationSize is set to be larger than the array-size
    intercept[IllegalArgumentException] {
      LIBLINEAR.takeAbsoluteBinaryStratisfied(smallTestData, 20)
    }
  }
}