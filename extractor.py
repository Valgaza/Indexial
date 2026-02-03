import pymupdf4llm
import fitz
import re
from pathlib import Path

def detect_dynamic_margins(pdf_path, scan_pages=5, threshold_y_top=100, threshold_y_bottom=100):
    """
    Scans pages to find RECURRING headers/footers.
    Only defines a margin if text actually repeats or looks like a page number.
    """
    doc = fitz.open(pdf_path)
    if len(doc) < 2:
        return (0, 0, 0, 0) # Single page docs don't have "running" headers

    page_height = doc[0].rect.height
    pages_to_scan = min(len(doc), scan_pages)
    
    # Dictionaries to track text occurrence: { "text_content": [list_of_y_positions] }
    top_zone_text = {}
    bottom_zone_text = {}

    for i in range(pages_to_scan):
        page = doc[i]
        blocks = page.get_text("blocks")
        
        for b in blocks:
            y0, y1 = b[1], b[3]
            text = b[4].strip()
            if not text: continue
            
            # Identify candidates
            if y1 < threshold_y_top:
                if text not in top_zone_text: top_zone_text[text] = []
                top_zone_text[text].append(y1)
                
            if y0 > (page_height - threshold_y_bottom):
                if text not in bottom_zone_text: bottom_zone_text[text] = []
                bottom_zone_text[text].append(y0)

    # --- Analysis Helper ---
    def get_safe_margin(candidates_dict, is_top=True):
        max_cut = 0
        min_cut = float('inf')
        found_artifact = False

        for text, positions in candidates_dict.items():
            # RULE 1: Repetition
            # It's a header if it appears on at least 50% of the scanned pages (min 2)
            is_repeating = len(positions) >= max(3, pages_to_scan * 0.5)
            
            # RULE 2: Pattern Matching (Page numbers)
            # Regex for "1", "Page 1", "1 / 20", "- 1 -"
            is_page_num = re.search(r'^(page\s*)?(\d+|[ivx]+)(\s*/\s*\d+)?$|^-?\s*\d+\s*-?$', text, re.IGNORECASE)
            
            if is_repeating or is_page_num:
                found_artifact = True
                if is_top:
                    # For top, we want the lowest Y1 (bottom of the header text)
                    current_max = max(positions)
                    if current_max > max_cut: max_cut = current_max
                else:
                    # For bottom, we want the highest Y0 (top of the footer text)
                    current_min = min(positions)
                    if current_min < min_cut: min_cut = current_min

        if not found_artifact:
            return 0
        
        return max_cut if is_top else (page_height - min_cut)

    # --- Calculate Final Margins ---
    final_top = get_safe_margin(top_zone_text, is_top=True)
    final_bottom = get_safe_margin(bottom_zone_text, is_top=False)

    doc.close()

    # Add buffer only if we actually found something
    final_top = (final_top + 10) if final_top > 0 else 0
    final_bottom = (final_bottom + 10) if final_bottom > 0 else 0

    if final_top > 0 or final_bottom > 0:
        print(f"Dynamic Detection: Cropping Top {final_top:.1f}pts and Bottom {final_bottom:.1f}pts")
    else:
        print("Dynamic Detection: No headers/footers detected. Keeping full page.")

    return (0, final_top, 0, final_bottom)

def main():
    # 1. Define paths
    input_path = Path("Docs/trial.pdf")
    output_dir = Path("output/markdown")
    output_file = output_dir / "trial.md"

    # 2. Check input
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # 3. Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 4. Calculate margins safely
        calculated_margins = detect_dynamic_margins(str(input_path))
        
        # 5. Convert
        md_text = pymupdf4llm.to_markdown(
            str(input_path),
            margins=calculated_margins
        )
        
        output_file.write_text(md_text, encoding="utf-8")
        print(f"Successfully converted '{input_path}' to '{output_file}'")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    main()