import java.nio.file.{Files, Paths}
import scala.collection.mutable

val file_path = java.nio.file.Paths.get("").toAbsolutePath
println (file_path)

val in_path = file_path.resolve ("data/error_func/func_before")
val out_path = file_path.resolve("data/error_func/full_dot_before")

importCode (
    inputPath = in_path.toString,
    projectName = "error_funcs_before"
)

cpg.method.l.foreach { m =>
    val method_name = m.name
    val file_name = m.file.name.headOption.getOrElse("unknown.cpp")
    val dot_name = file_name.replace(".cpp", ".dot")
    val dot_path = out_path.resolve(dot_name)
    //val dot_text = m.dotAst.mkString("\n")

    val astMap: Map[String, joern.CodePropertyGraph#NodeType] =
      m.ast.l.map(n => n.id.toString -> n).toMap
  
    // 生成最终 DOT 文本
    val final_dot_text: String = m.dotAst.map { line =>
      val Pattern = """"(\d+)" \[label = <([^,]+), (\d+)<BR/>(.*)>]""".r
  
      line match {
        case Pattern(id, labelType, lineNo, _) =>
          val fullCode: String = astMap.get(id).flatMap(n => Option(n.code)).getOrElse("")
          val escCode: String = fullCode
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("\n", "<BR/>")
  
          s""""$id" [label=<$labelType, $lineNo<BR/>$escCode>]"""
  
        case _ =>
          line // 不匹配的原样保留
      }
    }.mkString("\n")




    Files.write(dot_path, final_dot_text.getBytes("UTF-8"))
    println(s"[OK] AST dot generate: $dot_path")
}
