importCode(inputPath="./data/correct_func/func/0.cpp", projectName="0.cpp")
workspace
//cpg.method.name.l
//cpg.method.name("check_rodc_critical_attribute").astChildren.l
//cpg.method.name("check_rodc_critical_attribute").dotAst.foreach(println)

cpg.method.l.foreach { m =>
    m.ast.l.foreach { n =>
        val t = n match {
            case x if x.propertyOption ("TYPE_FULL_NAME").isDefined => 
                x.property ("TYPE_FULL_NAME").toString
            case _ => ""
        }
        println (
            s"id = ${n.id}, label = ${n.label}, code = ${n.code}, type = ${t}"
        )
    }
}

//var dotText = cpg.method.name("check_rodc_critical_attribute").dotAst.mkString("\n")
//import java.nio.file.{Files,Paths}
//Files.write(Paths.get("check_rodc_critical_attribute.dot"), dotText.getBytes("UTF-8"))

