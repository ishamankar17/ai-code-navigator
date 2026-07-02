# tests/test_embedder.py
#
# Tests for the code-chunk extraction logic in embedder.py. These deliberately
# cover the edge cases we hit in real usage:
#   - the TypeScript return-type annotation bug (": string {" was not matched)
#   - the comment-offset bug in the Java extractor (chunks sliced from the
#     wrong position after comments were stripped)
#   - the known overlap behavior (a class chunk AND its inner method chunk
#     both being returned) so future changes don't silently change this
#     without us noticing
import pytest

from embedder import (
    extract_functions_from_python,
    extract_functions_from_js_ts,
    extract_functions_from_java,
    extract_functions_from_file,
    SUPPORTED_EXTS,
)


# ---------------------------------------------------------------------------
# Python (ast-based)
# ---------------------------------------------------------------------------

class TestPythonExtractor:
    def test_extracts_top_level_function(self):
        source = "def add(a, b):\n    return a + b\n"
        chunks = extract_functions_from_python(source)
        assert len(chunks) == 1
        assert "def add(a, b):" in chunks[0]

    def test_extracts_class_and_methods_separately(self):
        source = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n"
        )
        chunks = extract_functions_from_python(source)
        # ast.walk visits the ClassDef AND each method FunctionDef separately,
        # so we expect 3 chunks: the class, and each of its two methods.
        assert len(chunks) == 3
        joined = "\n".join(chunks)
        assert "class Calculator:" in joined
        assert "def add(self, a, b):" in joined
        assert "def subtract(self, a, b):" in joined

    def test_extracts_async_function(self):
        source = "async def fetch_data():\n    return await get()\n"
        chunks = extract_functions_from_python(source)
        assert len(chunks) == 1
        assert "async def fetch_data" in chunks[0]

    def test_no_functions_returns_empty_list(self):
        source = "x = 1\ny = 2\nprint(x + y)\n"
        chunks = extract_functions_from_python(source)
        assert chunks == []

    def test_invalid_syntax_raises(self):
        # extract_functions_from_python uses ast.parse, which raises
        # SyntaxError on invalid input. Callers (embed_code_chunks) are
        # expected to catch this per-file.
        source = "def broken(:\n    pass\n"
        with pytest.raises(SyntaxError):
            extract_functions_from_python(source)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript (brace-matching)
# ---------------------------------------------------------------------------

class TestJsTsExtractor:
    def test_plain_js_function_declaration(self):
        source = "function add(a, b) {\n  return a + b;\n}\n"
        chunks = extract_functions_from_js_ts(source)
        assert len(chunks) == 1
        assert chunks[0].startswith("function add(a, b)")

    def test_plain_js_arrow_function(self):
        source = "const double = (x) => {\n  return x * 2;\n};\n"
        chunks = extract_functions_from_js_ts(source)
        assert len(chunks) == 1
        assert "const double = (x) =>" in chunks[0]

    def test_plain_js_class_and_method(self):
        source = (
            "class Plain {\n"
            "  method(x) {\n"
            "    return x;\n"
            "  }\n"
            "}\n"
        )
        chunks = extract_functions_from_js_ts(source)
        # Known overlap behavior: the class body and its inner method are
        # both captured as separate chunks (this is handled downstream by
        # retriever.py's deduplication, not by the extractor itself).
        assert len(chunks) == 2
        assert any(c.startswith("class Plain") for c in chunks)
        assert any(c.startswith("method(x)") for c in chunks)

    def test_typescript_typed_function_return_type(self):
        # Regression test: this exact snippet failed before the fix because
        # the regex required `{` immediately after `)`, with no allowance
        # for a TypeScript return-type annotation in between.
        source = (
            "function getUserName(user: User): string {\n"
            "  return user.fullName;\n"
            "}\n"
        )
        chunks = extract_functions_from_js_ts(source)
        assert len(chunks) == 1
        assert chunks[0].startswith("function getUserName(user: User): string {")

    def test_typescript_typed_arrow_function(self):
        source = "const add = (a: number, b: number): number => {\n  return a + b;\n};\n"
        chunks = extract_functions_from_js_ts(source)
        assert len(chunks) == 1
        assert "const add = (a: number, b: number): number =>" in chunks[0]

    def test_typescript_generic_function(self):
        source = "function identity<T>(value: T): T {\n  return value;\n}\n"
        chunks = extract_functions_from_js_ts(source)
        assert len(chunks) == 1
        assert "function identity<T>(value: T): T {" in chunks[0]

    def test_typescript_generic_class_with_typed_async_method(self):
        source = (
            "class Repo<T> {\n"
            "  private items: T[] = [];\n"
            "\n"
            "  add(item: T): void {\n"
            "    this.items.push(item);\n"
            "  }\n"
            "\n"
            "  async fetchAll(): Promise<T[]> {\n"
            "    return this.items;\n"
            "  }\n"
            "}\n"
        )
        chunks = extract_functions_from_js_ts(source)
        joined = "\n".join(chunks)
        # Regression: "class Repo<T> {" was not matched at all before the fix,
        # because the class pattern had no generics group (only function
        # patterns did), so a generic class's own chunk was silently dropped.
        assert "class Repo<T> {" in joined
        assert "add(item: T): void {" in joined
        assert "async fetchAll(): Promise<T[]> {" in joined

    def test_no_functions_returns_empty_list(self):
        source = "const x = 1;\nconst y = 2;\nconsole.log(x + y);\n"
        chunks = extract_functions_from_js_ts(source)
        assert chunks == []

    def test_nested_braces_do_not_truncate_chunk(self):
        # The brace counter must handle nested blocks (if/for/object literals)
        # correctly, or the chunk gets cut off at the first inner "}".
        source = (
            "function outer(x) {\n"
            "  if (x > 0) {\n"
            "    for (let i = 0; i < x; i++) {\n"
            "      console.log(i);\n"
            "    }\n"
            "  }\n"
            "  return { done: true };\n"
            "}\n"
        )
        chunks = extract_functions_from_js_ts(source)
        # Regression: the generic "method-like" pattern (\w+\(...\)\s*{) used
        # to also match "if (...)" and "for (...)" as if they were method
        # calls, since they share identifier(...) { syntax. That produced 3
        # chunks (the real function plus two false-positive control-flow
        # "methods") instead of just the one real function.
        assert len(chunks) == 1
        assert chunks[0].count("{") == chunks[0].count("}")
        assert "return { done: true };" in chunks[0]

    def test_control_flow_keywords_not_matched_as_standalone_methods(self):
        # Direct test of the false-positive bug: if/for/while/switch/catch
        # blocks at the top level (not inside any function) must not be
        # extracted as if they were method definitions.
        source = (
            "if (x > 0) {\n"
            "  doSomething();\n"
            "}\n"
            "for (let i = 0; i < 10; i++) {\n"
            "  doSomethingElse();\n"
            "}\n"
            "while (true) {\n"
            "  break;\n"
            "}\n"
        )
        chunks = extract_functions_from_js_ts(source)
        assert chunks == []


# ---------------------------------------------------------------------------
# Java (brace-matching with comment stripping)
# ---------------------------------------------------------------------------

class TestJavaExtractor:
    def test_simple_class_and_method(self):
        source = (
            "public class Calculator {\n"
            "    public int add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        joined = "\n".join(chunks)
        assert "public class Calculator {" in joined
        assert "public int add(int a, int b) {" in joined

    def test_interface_and_enum(self):
        source = (
            "interface Greeter {\n"
            "    String greet(String name);\n"
            "}\n"
            "\n"
            "enum Status {\n"
            "    ACTIVE, INACTIVE;\n"
            "\n"
            "    public boolean isActive() {\n"
            "        return this == ACTIVE;\n"
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        joined = "\n".join(chunks)
        assert "interface Greeter {" in joined
        assert "enum Status {" in joined
        assert "public boolean isActive() {" in joined

    def test_block_comment_does_not_shift_chunk_offsets(self):
        # Regression test: comments must be replaced with equal-length
        # whitespace (not deleted) when building the comment-stripped copy
        # used for matching, otherwise character offsets between the cleaned
        # text and the original source diverge and chunks get sliced from
        # the wrong position.
        source = (
            "/**\n"
            " * A sample service class.\n"
            " */\n"
            "public class UserService {\n"
            "    public String toString() {\n"
            "        // simple debug string\n"
            '        return "UserService";\n'
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        joined = "\n".join(chunks)
        # The class chunk must start exactly at "public class", not mid-comment.
        class_chunks = [c for c in chunks if "class UserService" in c]
        assert len(class_chunks) == 1
        assert class_chunks[0].startswith("public class UserService {")
        assert "public String toString() {" in joined

    def test_line_comment_does_not_shift_chunk_offsets(self):
        source = (
            "public class Foo { // trailing comment\n"
            "    public void bar() {\n"
            "        doSomething();\n"
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        bar_chunks = [c for c in chunks if c.startswith("public void bar()")]
        assert len(bar_chunks) == 1

    def test_method_with_throws_clause(self):
        source = (
            "public class Validator {\n"
            "    private boolean isValid(String name) throws IllegalArgumentException {\n"
            "        if (name == null) {\n"
            '            throw new IllegalArgumentException("null name");\n'
            "        }\n"
            "        return name.length() > 0;\n"
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        method_chunks = [c for c in chunks if "isValid" in c and c.startswith("private boolean")]
        assert len(method_chunks) == 1
        assert "throws IllegalArgumentException" in method_chunks[0]

    def test_annotated_method(self):
        source = (
            "public class Foo {\n"
            "    @Override\n"
            "    public String toString() {\n"
            '        return "Foo";\n'
            "    }\n"
            "}\n"
        )
        chunks = extract_functions_from_java(source)
        joined = "\n".join(chunks)
        assert "public String toString()" in joined

    def test_no_functions_returns_empty_list(self):
        source = "package com.example;\n\nimport java.util.List;\n"
        chunks = extract_functions_from_java(source)
        assert chunks == []


# ---------------------------------------------------------------------------
# File-extension routing
# ---------------------------------------------------------------------------

class TestExtractFunctionsFromFile:
    @pytest.mark.parametrize("ext", [".py", ".js", ".jsx", ".ts", ".tsx", ".java"])
    def test_supported_extensions_are_routed(self, ext, tmp_path):
        assert ext in SUPPORTED_EXTS

        if ext == ".py":
            content = "def f():\n    pass\n"
        elif ext == ".java":
            content = "public class F {\n    public void f() {\n    }\n}\n"
        else:
            content = "function f() {\n  return 1;\n}\n"

        file_path = tmp_path / f"sample{ext}"
        file_path.write_text(content, encoding="utf-8")

        chunks = extract_functions_from_file(str(file_path))
        assert len(chunks) >= 1

    def test_unsupported_extension_returns_empty_list(self, tmp_path):
        file_path = tmp_path / "sample.rb"
        file_path.write_text("def f\n  1\nend\n", encoding="utf-8")
        chunks = extract_functions_from_file(str(file_path))
        assert chunks == []