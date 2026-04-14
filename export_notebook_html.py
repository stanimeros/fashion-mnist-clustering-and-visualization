#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


PRINT_CSS = """
<style id="print-layout-tweaks">
@page { size: auto; margin: 0.8in; }
body {
  margin: 0 auto !important;
  max-width: 8.0in;
  padding-left: 0 !important;
  padding-right: 0 !important;
}
.input_prompt, .output_prompt, .jp-InputPrompt, .jp-OutputPrompt {
  display: none !important;
}
@media print {
  body {
    margin: 0 auto !important;
    max-width: 100% !important;
  }
  .input_prompt, .output_prompt, .jp-InputPrompt, .jp-OutputPrompt {
    display: none !important;
  }
}
</style>
""".strip()


def inject_print_css(html: str) -> str:
    if 'id="print-layout-tweaks"' in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{PRINT_CSS}\n</head>", 1)
    return f"{PRINT_CSS}\n{html}"


def main() -> None:
    root = Path.cwd()
    notebook_path = root / "results_report.ipynb"
    output_path = root / "results_report.html"

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    client.execute()
    nbformat.write(nb, notebook_path)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _resources = exporter.from_notebook_node(nb)
    html = inject_print_css(html)
    output_path.write_text(html, encoding="utf-8")

    print(f"Updated notebook: {notebook_path.name}")
    print(f"Created HTML: {output_path.name}")


if __name__ == "__main__":
    main()
