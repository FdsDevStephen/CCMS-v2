from pathlib import Path


def get_case_number_from_filename(pdf_path: str | Path) -> str:
    """
    Returns the filename without extension.

    Example:
        WP-19110-2024-B.pdf
        ->
        WP-19110-2024-B
    """

    return Path(pdf_path).stem