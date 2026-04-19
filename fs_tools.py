import os
import re
import datetime
from pathlib import Path


def read_file(filepath: str, **kwargs) -> dict:
    """Read resume files (PDF, TXT, DOCX) and return structured response."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return {"success": False, "error": f"File not found: {filepath}"}

        ext = filepath.suffix.lower()
        content = ""

        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(filepath))
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return {"success": False, "error": "PyPDF2 not installed. Run: pip install PyPDF2"}

        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(str(filepath))
                content = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return {"success": False, "error": "python-docx not installed. Run: pip install python-docx"}

        else:
            return {"success": False, "error": f"Unsupported file type: {ext}"}

        stat = filepath.stat()
        return {
            "success": True,
            "content": content,
            "metadata": {
                "filename": filepath.name,
                "extension": ext,
                "size_bytes": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_files(directory: str, extension: str = None, **kwargs) -> list:
    """List all files in a directory, optionally filtered by extension."""
    try:
        directory = Path(directory)
        if not directory.exists():
            return [{"success": False, "error": f"Directory not found: {directory}"}]

        # Ignore empty or invalid extensions
        if extension and extension.strip() in ("", "."):
            extension = None

        # Handle comma-separated extensions
        ext_list = None
        if extension:
            ext_list = [e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in extension.split(",")]

        files = []
        for item in directory.iterdir():
            if not item.is_file():
                continue
            if ext_list:
                if item.suffix.lower() not in [e.lower() for e in ext_list]:
                    continue
            stat = item.stat()
            files.append({
                "name": item.name,
                "path": str(item),
                "size_bytes": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return files

    except Exception as e:
        return [{"success": False, "error": str(e)}]


def write_file(filepath: str, content: str, **kwargs) -> dict:
    """Write content to file, creating directories if needed."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "message": f"File written: {filepath}", "size_bytes": filepath.stat().st_size}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_in_file(filepath: str, keyword: str, **kwargs) -> dict:
    """Search for keyword in file content with surrounding context."""
    result = read_file(filepath)
    if not result.get("success"):
        return result

    content = result["content"]
    lines = content.split("\n")
    matches = []

    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            context_start = max(0, i - 1)
            context_end = min(len(lines), i + 2)
            matches.append({
                "line_number": i + 1,
                "line": line.strip(),
                "context": "\n".join(lines[context_start:context_end]),
            })

    return {
        "success": True,
        "filepath": filepath,
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
    }