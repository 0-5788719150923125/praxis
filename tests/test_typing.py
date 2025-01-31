# import ast
# import inspect
# from pathlib import Path
# from typing import List, NamedTuple, Set

# import pytest


# class UntypedFunction(NamedTuple):
#     name: str
#     lineno: int
#     signature: str
#     context: str


# class CodeAnalyzer(ast.NodeVisitor):
#     def __init__(self, source_code: str):
#         self.source_code = source_code.splitlines()
#         self.functions: List[ast.FunctionDef] = []
#         self.typed_functions: Set[ast.FunctionDef] = set()

#     def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
#         self.functions.append(node)

#         # Check if this is a method
#         is_method = node.args.args and node.args.args[0].arg == "self"

#         # For methods, skip the first argument (self)
#         args_to_check = node.args.args[1:] if is_method else node.args.args

#         # Check return type and remaining arguments
#         if node.returns or any(arg.annotation for arg in args_to_check):
#             self.typed_functions.add(node)

#     def get_untyped_functions(self) -> List[UntypedFunction]:
#         untyped = []
#         for func in set(self.functions) - self.typed_functions:
#             # Get the function's source lines
#             start_line = func.lineno - 1
#             end_line = (
#                 func.end_lineno if hasattr(func, "end_lineno") else start_line + 1
#             )
#             context = "\n".join(self.source_code[start_line:end_line])

#             # Build function signature
#             args = [arg.arg for arg in func.args.args]
#             signature = f"def {func.name}({', '.join(args)})"

#             untyped.append(
#                 UntypedFunction(
#                     name=func.name,
#                     lineno=func.lineno,
#                     signature=signature,
#                     context=context,
#                 )
#             )
#         return sorted(untyped, key=lambda x: x.lineno)


# @pytest.fixture
# def code_analyzer():
#     def _analyze_file(file_path: Path) -> CodeAnalyzer:
#         with open(file_path) as f:
#             source = f.read()
#             tree = ast.parse(source)
#             analyzer = CodeAnalyzer(source)
#             analyzer.visit(tree)
#             return analyzer

#     return _analyze_file


# def get_python_files(start_path: Path) -> List[Path]:
#     return list(start_path.rglob("*.py"))


# @pytest.mark.parametrize("py_file", get_python_files(Path("praxis")))
# def test_type_hints_present(py_file: Path, code_analyzer: CodeAnalyzer) -> None:
#     analyzer = code_analyzer(py_file)
#     untyped = analyzer.get_untyped_functions()

#     if untyped:
#         error_msg = [f"\nType hints missing in {py_file}:"]
#         for func in untyped:
#             error_msg.extend(
#                 [
#                     f"\n{'='*50}",
#                     f"Function: {func.name}",
#                     f"Line: {func.lineno}",
#                     f"Signature: {func.signature}",
#                     f"Context:",
#                     # f"{func.context}",
#                 ]
#             )
#         pytest.fail("\n".join(error_msg))
