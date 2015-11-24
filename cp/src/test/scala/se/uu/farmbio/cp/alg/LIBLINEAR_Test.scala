package se.uu.farmbio.cp.alg

import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.junit.runner.RunWith

@RunWith(classOf[JUnitRunner])
class LIBLINEAR_Test extends FunSuite{
  
  test("util function takeFractions"){
    val origList = (1 to 50).toList;
    
    // equal partitions:
    val outputLists = LIBLINEAR.takeFractions(origList, Array(100,100,100,100,100));
    assert(outputLists.length == 5, 
        "The number of created sub-lists should correspond to the given number");
    outputLists.foreach {subList => assert(subList.length==10, "Each sub-list should be correct length") }
    assert(origList.intersect(outputLists.flatten.toSeq).length == origList.length, 
        "The same numbers should be in the outputList as in the original list")
    
    // un-equal partitions:
    val unequalLists = LIBLINEAR.takeFractions(origList, Array(5,15,25,5))
    assert(unequalLists.length == 4)
    assert(unequalLists(0).size == 5)
    assert(unequalLists(1).size == 15)
    assert(unequalLists(2).size == 25)
    assert(unequalLists(3).size == 5)
    
    // when fractions is not perfectly divisible 
    val dontAddUp = LIBLINEAR.takeFractions(origList, Array(100,100,100))
    assert(dontAddUp.length == 3)
    assert(dontAddUp.flatten.length == origList.length, 
        "When fractions are not perfectly dividing the input-list, should not be a problem")
    
    //corner-case: 0 or 1 fraction specified 
    val smallList=(1 to 5).toList;
    val outSmall = LIBLINEAR.takeFractions(smallList, Array())
    assert(! outSmall(0).equals(smallList), 
        "In cornercase (0 or 1 fractions given), the output should be randomized")
    val output0Fracs = LIBLINEAR.takeFractions(smallList, Array())
    
    assert(output0Fracs.length==1, 
        "0 fracs should just return a randomized list");
    assert(output0Fracs(0).length == smallList.size);
    assert(LIBLINEAR.takeFractions(smallList, Array(100)).length==1, 
        "");
    
    //corner-case: more than 10 fractions specified (should throw exception)
    intercept[IllegalArgumentException]{
      LIBLINEAR.takeFractions(List(), (1 to 11).toList.map(_.toDouble).toArray)
    }
    
    // fractions should all be positive
    intercept[IllegalArgumentException]{
      LIBLINEAR.takeFractions(List(), Array(-1,0, 50,1))
    }
    // sum of all fractions must be greater than 0
    intercept[IllegalArgumentException]{
      LIBLINEAR.takeFractions(List(), Array(0,0))
    }
  }
}