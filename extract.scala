import java.nio.file.{Files, Paths}

val file_path = java.nio.file.Paths.get("").toAbsolutePath
println (file_path)

val in_path = file_path.resolve ("data/error_func/func_before")
val out_path = file_path.resolve("data/error_func/dot_before")

importCode (
    inputPath = in_path.toString,
    projectName = "error_funcs_before"
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
