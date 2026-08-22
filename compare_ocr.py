"""
Parity + timing comparison: ocr.py (v1) vs ocr_v2.py (v2).

v1 note: process_document() reads module globals that are only ever set by
OCRProcessor._configure(), which runs inside .process() -- not __init__.
Calling process_document() directly raises NameError on POPPLER_PATH, so we
set it here, exactly as _configure() would.
"""

import contextlib
import io
import time
from pathlib import Path

POPPLER = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime\dependencies\native\poppler\Library\bin"
)

import ocr as V1
import ocr_v2 as V2

V1.POPPLER_PATH = POPPLER

rows = []

for pdf in sorted(Path("uploads").glob("*.pdf")):

    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):

        t = time.perf_counter()
        r1 = V1.process_document(pdf)
        t1 = time.perf_counter() - t

        t = time.perf_counter()
        r2 = V2.process_document(pdf)
        t2 = time.perf_counter() - t

    page = lambda r: (r["locations"]["prayer"] or {}).get("page")

    rows.append(
        (
            pdf.stem,
            page(r1),
            page(r2),
            len(r1["sections"]["prayer"]),
            len(r2["sections"]["prayer"]),
            len(r1["sections"]["pages_2_to_13"]),
            len(r2["sections"]["body"]),
            t1,
            t2,
            len(r2["text_layer_pages"]),
        )
    )

    print(f"done {pdf.stem}", flush=True)


print()
print(
    f'{"PDF":26}{"v1pg":>5}{"v2pg":>5}'
    f'{"v1pray":>8}{"v2pray":>8}'
    f'{"v1body":>8}{"v2body":>8}'
    f'{"TL":>4}{"v1s":>7}{"v2s":>7}'
)
print("-" * 86)

sum1 = sum2 = 0.0
diffs = []

for name, p1, p2, pr1, pr2, b1, b2, t1, t2, tl in rows:

    flag = ""

    if p1 != p2:
        flag = "  <-- PAGE DIFF"
        diffs.append(name)

    print(
        f"{name:26}{str(p1):>5}{str(p2):>5}"
        f"{pr1:>8}{pr2:>8}{b1:>8}{b2:>8}"
        f"{tl:>4}{t1:>7.1f}{t2:>7.1f}{flag}"
    )

    sum1 += t1
    sum2 += t2

print("-" * 86)
print(f'{"TOTAL":26}{"":>5}{"":>5}{"":>8}{"":>8}{"":>8}{"":>8}{"":>4}{sum1:>7.1f}{sum2:>7.1f}')
print()
print(f"overall speedup : {sum1 / sum2:.2f}x")
print(f"prayer page diffs: {len(diffs)}/{len(rows)}  {diffs}")

found1 = sum(1 for r in rows if r[1] is not None)
found2 = sum(1 for r in rows if r[2] is not None)
print(f"prayer found     : v1={found1}/{len(rows)}  v2={found2}/{len(rows)}")

empty1 = sum(1 for r in rows if r[3] == 0)
empty2 = sum(1 for r in rows if r[4] == 0)
print(f"empty prayer text: v1={empty1}/{len(rows)}  v2={empty2}/{len(rows)}")
