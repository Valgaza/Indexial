import pymupdf4llm
from pathlib import Path

def main():
    # 1. Define paths clearly
    input_path = Path("Docs/trial.pdf")
    output_dir = Path("output/markdown")
    output_file = output_dir / "trial.md"

    # 2. Check if input exists BEFORE processing
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # 3. Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 4. Convert PDF to Markdown (using string path for compatibility)
        md_text = pymupdf4llm.to_markdown(str(input_path))
        
        # 5. Write the result to the file
        output_file.write_text(md_text, encoding="utf-8")
        print(f"Successfully converted '{input_path}' to '{output_file}'")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    main()
