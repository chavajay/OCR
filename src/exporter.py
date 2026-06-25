"""Result export module for OCR output.

Formats recognized text with proper line breaks and exports to
plain text (.txt) and Markdown (.md) formats.
"""

import os


class ResultExporter:
    """Exports OCR-recognized text to structured file formats.

    Takes a list of lines (each line is a string of recognized
    characters) and writes them to .txt or .md files preserving
    the original document layout.
    """

    def __init__(self, output_dir: str = "output"):
        """Initializes the exporter with an output directory.

        Args:
            output_dir: Directory path where exported files will be
                        written. Created automatically if it does not
                        exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def to_text(self, lines: list[str], filename: str = "output.txt") -> str:
        """Exports recognized text lines to a plain text file.

        Each line in the input list is written as a separate line
        in the output file, preserving paragraph structure.

        Args:
            lines: List of strings, each representing a recognized
                   text line from the document.
            filename: Output filename (will be placed in output_dir).

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If lines list is empty.
            IOError: If file write fails.
        """
        if not lines:
            raise ValueError("Cannot export empty text. Provide at least one line.")

        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except IOError as e:
            raise IOError(f"Failed to write text file {filepath}: {e}")

        print(f"Text exported to {filepath}")
        return os.path.abspath(filepath)

    def to_markdown(self, lines: list[str], filename: str = "output.md") -> str:
        """Exports recognized text lines to a Markdown file.

        Each line is written preserving line breaks. No extra
        formatting is applied beyond the raw text content.

        Args:
            lines: List of strings, each representing a recognized
                   text line from the document.
            filename: Output filename (will be placed in output_dir).

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If lines list is empty.
            IOError: If file write fails.
        """
        if not lines:
            raise ValueError("Cannot export empty text. Provide at least one line.")

        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write("title: OCR Output\n")
                f.write("---\n\n")
                for i, line in enumerate(lines):
                    f.write(line + "\n")
                    if i < len(lines) - 1 and line.strip() == "":
                        f.write("\n")
        except IOError as e:
            raise IOError(f"Failed to write Markdown file {filepath}: {e}")

        print(f"Markdown exported to {filepath}")
        return os.path.abspath(filepath)

    def export(
        self,
        lines: list[str],
        txt_filename: str = "output.txt",
        md_filename: str = "output.md",
    ) -> dict[str, str]:
        """Exports recognized text to both .txt and .md formats.

        Args:
            lines: List of recognized text lines.
            txt_filename: Name for the .txt file.
            md_filename: Name for the .md file.

        Returns:
            Dictionary with keys 'txt' and 'md' mapping to absolute
            file paths.
        """
        return {
            "txt": self.to_text(lines, filename=txt_filename),
            "md": self.to_markdown(lines, filename=md_filename),
        }
