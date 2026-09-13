import base64
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MistralOCRExtractor:
    """
    Mistral Document AI OCR-based PDF to Markdown extractor.
    Uses the dedicated /v1/ocr endpoint for document processing.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("MISTRAL_OCR")
        if not self.api_key:
            raise ValueError("MISTRAL_OCR API key not found in environment variables")

        # Mistral OCR endpoint
        self.api_url = "https://api.mistral.ai/v1/ocr"

    def pdf_to_base64(self, pdf_path):
        """
        Convert PDF file to base64 data URL.

        Args:
            pdf_path: Path to PDF file

        Returns:
            str: Data URL with base64-encoded PDF
        """
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        return f"data:application/pdf;base64,{pdf_base64}"

    def ocr_pdf(self, pdf_path, max_retries=3):
        """
        Send PDF to Mistral OCR API and get markdown text with retry logic.

        Args:
            pdf_path: Path to PDF file
            max_retries: Maximum number of retry attempts

        Returns:
            dict: OCR response with pages containing markdown
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Convert PDF to base64 data URL
        file_data_url = self.pdf_to_base64(pdf_path)

        # Build request payload
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": file_data_url
            },
            "table_format": "markdown",  # Can be null, "markdown", or "html"
            "include_image_base64": False  # Set to true if you need image data
        }

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{max_retries}...")
                else:
                    print(f"Sending OCR request for {Path(pdf_path).name}...")

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=180  # 3 minute timeout for large PDFs
                )

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                result = response.json()

                # Check for pages in response
                if "pages" in result and len(result["pages"]) > 0:
                    print(f"✓ OCR completed successfully - {len(result['pages'])} page(s)")
                    return result
                else:
                    print(f"Warning: Unexpected API response format")
                    print(f"Response: {result}")
                    return None

            except requests.exceptions.Timeout:
                print(f"Error: Request timeout for {Path(pdf_path).name}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                raise

            except requests.exceptions.RequestException as e:
                # Don't retry on 4xx errors (except 429)
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        print(f"Error: API request failed: {e}")
                        if hasattr(e.response, 'text'):
                            print(f"Response: {e.response.text}")
                        raise

                # Retry on 5xx errors or network issues
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    print(f"Error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                print(f"Error: API request failed after {max_retries} attempts: {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    print(f"Response: {e.response.text}")
                raise

            except Exception as e:
                print(f"Error: Unexpected error: {e}")
                raise

        # If we exhausted all retries
        raise Exception(f"Failed to process {Path(pdf_path).name} after {max_retries} attempts")

    def extract_pdf_to_markdown(self, pdf_path, output_path=None):
        """
        Extract text from PDF using Mistral OCR and save as markdown.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path to output markdown file (optional)

        Returns:
            str: Combined markdown text from all pages
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check file size (API limit: 50MB for PDF)
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            raise ValueError(f"PDF file too large: {file_size_mb:.1f}MB (max 50MB)")

        print(f"Opening PDF: {pdf_path} ({file_size_mb:.1f}MB)")

        # Send to OCR API
        result = self.ocr_pdf(pdf_path)

        if not result or "pages" not in result:
            raise Exception("OCR failed to return page results")

        # Extract markdown from all pages and inline tables
        markdown_pages = []
        for page in result["pages"]:
            page_markdown = page.get("markdown", "")

            # Replace table placeholders with actual table content
            if page_markdown and "tables" in page and page["tables"]:
                for table in page["tables"]:
                    table_id = table.get("id")
                    table_content = table.get("content", "")

                    if table_id and table_content:
                        # Replace placeholder like [tbl-0.md](tbl-0.md) with actual table
                        placeholder = f"[{table_id}]({table_id})"
                        page_markdown = page_markdown.replace(placeholder, table_content)

            if page_markdown:
                markdown_pages.append(page_markdown)

        # Combine all pages with separators
        combined_markdown = "\n\n---\n\n".join(markdown_pages)

        # Print stats
        if "usage_info" in result:
            pages_processed = result["usage_info"].get("pages_processed", "unknown")
            doc_size = result["usage_info"].get("doc_size_bytes", 0)
            print(f"Processed {pages_processed} page(s), {doc_size / 1024:.1f}KB")

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(combined_markdown, encoding="utf-8")
            print(f"✓ Saved markdown to: {output_path}")

        return combined_markdown


def process_all_pdfs(input_dir="Docs", output_dir="output/markdown"):
    """
    Process all PDF files in the input directory.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save markdown files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")

    # Initialize extractor
    try:
        extractor = MistralOCRExtractor()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Process each PDF (with rate limiting between files)
    for idx, pdf_file in enumerate(pdf_files):
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name} ({idx + 1}/{len(pdf_files)})")
        print(f"{'='*60}")

        # Generate output filename
        output_file = output_path / f"{pdf_file.stem}.md"

        try:
            extractor.extract_pdf_to_markdown(pdf_file, output_file)
            print(f"✓ Successfully converted: {pdf_file.name}")

            # Add delay between files to avoid rate limits (except for last file)
            if idx < len(pdf_files) - 1:
                print(f"Waiting 3 seconds before next file...")
                time.sleep(3)

        except Exception as e:
            print(f"✗ Failed to convert {pdf_file.name}: {e}")
            continue


def main():
    """
    Main entry point - processes all PDFs in Docs/ folder.
    """
    process_all_pdfs(input_dir="Docs", output_dir="output/markdown")


if __name__ == "__main__":
    main()
