import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, Set


def extract_imports(file_path: str) -> Set[str]:
    """Extract all import statements from a Python file, ignoring relative imports."""
    imports = set()

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Parse the Python file
        tree = ast.parse(content)

        for node in ast.walk(tree):
            # Handle 'import x' statements
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Only add if not a relative import
                    if not name.name.startswith("."):
                        imports.add(name.name.split(".")[0])

            # Handle 'from x import y' statements
            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (those with level > 0 or starting with dots)
                if node.module and not node.level > 0:
                    imports.add(node.module.split(".")[0])

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

    return imports


def find_python_files(root_dir: str) -> Dict[str, Set[str]]:
    """Recursively find all Python files and their imports."""
    python_files_imports = {}

    # Walk through all directories
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                imports = extract_imports(file_path)

                # Filter out standard library imports
                third_party_imports = {
                    imp for imp in imports if not is_standard_library(imp)
                }

                if third_party_imports:
                    python_files_imports[relative_path] = third_party_imports

    return python_files_imports


def is_standard_library(module_name: str) -> bool:
    """Check if a module is part of Python's standard library."""
    import distutils.sysconfig as sysconfig

    std_lib = sysconfig.get_python_lib(standard_lib=True)

    # Check if module exists in standard library path
    for path in sys.path:
        if path.startswith(std_lib):
            if os.path.exists(os.path.join(path, module_name)) or os.path.exists(
                os.path.join(path, module_name + ".py")
            ):
                return True
    return False


def main():
    # Replace with your project root directory
    project_root = sys.argv[1]

    # Get all imports
    all_imports = find_python_files(project_root)

    # Aggregate all unique third-party imports
    unique_imports = set()
    for imports in all_imports.values():
        unique_imports.update(imports)

    # Print results
    print("\nFound imports by file:")
    for file_path, imports in all_imports.items():
        print(f"\n{file_path}:")
        for imp in sorted(imports):
            print(f"  - {imp}")

    print("\nAll unique third-party imports:")
    for imp in sorted(unique_imports):
        print(f"- {imp}")

    # Optionally, write to requirements-like file
    with open(f"{sys.argv[2]}_imports.txt", "w") as f:
        for imp in sorted(unique_imports):
            f.write(f"{imp}\n")


if __name__ == "__main__":
    main()
