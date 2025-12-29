importCode(inputPath="./data/correct_func/func_after", projectName="0.c")
workspace
cpg.method.name.l
cpg.method.name("check_rodc_critical_attribute").astChildren.l
cpg.method.name("check_rodc_critical_attribute").dotAst.foreach(println)

var dotText = cpg.method.name("check_rodc_critical_attribute").dotAst.mkString("\n")
import java.nio.file.{Files,Paths}
Files.write(Paths.get("check_rodc_critical_attribute.dot"), dotText.getBytes("UTF-8"))

