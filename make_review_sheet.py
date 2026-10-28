"""
=====================================================================
 Brain Tumor Detection - Label review sheet for an expert
 ---------------------------------------------------------------
 Collects every image whose label is in doubt into one printable HTML
 page with thumbnails, so a qualified reader can go through them quickly:

   1) decisions already in label_review.json (excluded / relabelled / kept)
   2) suspected label issues from cross-validation (cv_results.json):
      the model, which never saw the image, confidently disagrees
   3) conflict groups from duplicate detection (duplicates_report.json):
      the same scan stored under different labels (currently all dropped)

 Each item has an empty "Expert decision" field. Decisions are then
 entered in label_review.json (exclude / relabel / keep).

 The page embeds the images, so it is written to review/ (not tracked by
 git, the images belong to the datasets).

 Run (after train.py / cross_validate.py):
   python make_review_sheet.py
=====================================================================
"""

import base64
import html
import io
import json
import os

from PIL import Image

from config import (BASE_DIR, DATA_SOURCES, LABEL_REVIEW_PATH, CV_RESULTS_PATH,
                    DUPLICATES_REPORT_PATH)

OUTPUT_PATH = os.path.join(BASE_DIR, "review", "label_review_sheet.html")
THUMB = 260


def resolve(display):
    """'<source>/<path>' (see train.display_path) -> file path, or None."""
    source, _, rel = display.replace("\\", "/").partition("/")
    if source not in DATA_SOURCES:
        return None
    path = os.path.join(DATA_SOURCES[source]["dir"], rel)
    return path if os.path.exists(path) else None


def thumbnail(display):
    path = resolve(display)
    if path is None:
        return f'<div class="missing">not found:<br>{html.escape(display)}</div>'
    img = Image.open(path).convert("L")
    img.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" alt="{html.escape(display)}">'


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def item(title, paths, lines):
    images = "".join(f'<figure>{thumbnail(p)}<figcaption>{html.escape(p)}</figcaption></figure>'
                     for p in paths)
    details = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return (f'<section class="item"><h3>{html.escape(title)}</h3>'
            f'<div class="images">{images}</div><ul>{details}</ul>'
            '<p class="decision">Expert decision: &#9744; tumor &nbsp; &#9744; no tumor '
            '&nbsp; &#9744; unsure / exclude &nbsp;&nbsp; Notes: ______________________</p>'
            '</section>')


def main():
    sections = []

    review = load(LABEL_REVIEW_PATH) or {"images": []}
    items = [item(f"{d['action']}" + (f" -> {d['label']}" if d.get("label") else ""),
                  [d["path"]], [d.get("reason", "")]) for d in review["images"]]
    sections.append(("1. Current decisions in label_review.json", items))

    cv = load(CV_RESULTS_PATH) or {}
    items = [item(f"labelled '{s['label']}', model P(tumor) = {s['p_tumor']:.2f}",
                  [s["path"]], [f"fold {s['fold']} (the model never saw this image)"])
             for s in cv.get("suspected_label_issues", [])]
    sections.append(("2. Suspected label issues from cross-validation", items))

    dups = load(DUPLICATES_REPORT_PATH) or {}
    items = [item(f"same scan, {len(g)} copies with different labels", g,
                  ["labels follow the folder: " + ", ".join(p.split("/")[1] for p in g),
                   "currently: all copies dropped"])
             for g in dups.get("conflict_groups", [])]
    sections.append(("3. Same scan stored under different labels", items))

    body = "".join(f"<h2>{html.escape(title)} ({len(items)})</h2>"
                   + ("".join(items) or "<p>None.</p>") for title, items in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Review Sheet</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; background: #fff; }}
  h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 4px; margin-top: 32px; }}
  .item {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px 12px; margin: 12px 0;
           break-inside: avoid; }}
  .item h3 {{ margin: 0 0 8px; font-size: 1rem; }}
  .images {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  figure {{ margin: 0; max-width: {THUMB}px; }}
  figure img {{ max-width: 100%; display: block; background: #000; }}
  figcaption, .missing {{ font-size: .75rem; color: #555; word-break: break-all; }}
  .decision {{ margin: 8px 0 0; font-size: .9rem; }}
</style></head><body>
<h1>Label review sheet</h1>
<p>Images whose label is in doubt. These flags come from duplicate detection and
from a model, not from a diagnosis; please decide each case and enter the result
in <code>label_review.json</code> (exclude / relabel / keep).</p>
{body}
</body></html>"""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Review sheet with {sum(len(i) for _, i in sections)} items -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
