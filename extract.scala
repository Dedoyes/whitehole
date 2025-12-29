import java.nio.file.{Files, Paths}

val file_path = java.nio.file.Paths.get("").toAbsolutePath
println (file_path)

val in_path = file_path.resolve ("data/correct_func/func_after")
val out_path = file_path.resolve("data/correct_func/dot_after")

importCode (
    inputPath = in_path.toString,
    projectName = "correct_funcs_after"
)

cpg.method.l.foreach { m =>
    val method_name = m.name
    val file_name = m.file.name.headOption.getOrElse("unknown.cpp")
    val dot_name = file_name.replace(".cpp", ".dot")
    val dot_path = out_path.resolve(dot_name)
    val dot_text = m.dotAst.mkString("\n")
    Files.write(dot_path, dot_text.getBytes("UTF-8"))
    println(s"[OK] AST dot generate: $dot_path")
}
