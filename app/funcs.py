import os
from typing import List, Dict, Any
import pymupdf4llm


def parse_pdf(filepath: str, write_images: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a PDF file and extract text with metadata from each page using pymupdf4llm.
    
    Args:
        filepath (str): Path to the PDF file
        write_images (bool): Whether to extract and save images from the PDF
        
    Returns:
        list: List of dictionaries with format including filename, page, text, and additional metadata
    """
    result = []
    
    # Extract filename from filepath
    filename = os.path.basename(filepath)
    
    try:
        # Extract text using pymupdf4llm with page-wise extraction
        page_data_list = pymupdf4llm.to_markdown(
            filepath, 
            page_chunks = True,
            write_images = write_images
        )
        
        # Process each page's data
        for page_info in page_data_list:
            # Extract the text content
            page_text = page_info.get('text', '')
            page_metadata = page_info.get('metadata', {})
            
            # Create enhanced page data dictionary
            enhanced_page_data = {
                'filename': filename,
                'page': page_metadata.get('page', 0) + 1,  # Convert to 1-based indexing
                'text': page_text,
                'text_format': 'markdown',
                'extraction_method': 'pymupdf4llm',
                'has_tables': '|' in page_text,  # Basic table detection
                'char_count': len(page_text),
                'word_count': len(page_text.split()),
                'line_count': len(page_text.split('\n')),
                'images_extracted': write_images,
                'source_bbox': page_metadata.get('bbox', None),
                'source_page_size': page_metadata.get('page_size', None)
            }
            
            # Add any additional metadata from pymupdf4llm
            for key, value in page_metadata.items():
                if key not in ['page', 'bbox', 'page_size']:  # Avoid duplicates
                    enhanced_page_data[f'source_{key}'] = value
            
            result.append(enhanced_page_data)
                
    except Exception as e:
        print(f"Error parsing PDF {filepath}: {str(e)}")
        # Fallback: try without page chunks
        try:
            md_text_fallback = pymupdf4llm.to_markdown(filepath, write_images = write_images)
            page_data = {
                'filename': filename,
                'page': 1,
                'text': md_text_fallback,
                'text_format': 'markdown',
                'extraction_method': 'pymupdf4llm_fallback',
                'has_tables': '|' in md_text_fallback,
                'char_count': len(md_text_fallback),
                'word_count': len(md_text_fallback.split()),
                'line_count': len(md_text_fallback.split('\n')),
                'images_extracted': write_images,
                'error_note': 'Page-wise extraction failed, using full document'
            }
            result.append(page_data)
        except Exception as fallback_error:
            print(f"Fallback extraction also failed for {filepath}: {str(fallback_error)}")
            return []
    
    return result


def parse_pdfs(pdf_list: List[str], write_images: bool = False) -> List[Dict[str, Any]]:
    """
    Parse multiple PDF files and concatenate the results.
    
    Args:
        pdf_list (list): List of PDF file paths
        write_images (bool): Whether to extract and save images from the PDFs
        
    Returns:
        list: Concatenated list of dictionaries from all PDFs with enhanced metadata
    """
    all_results = []
    
    for pdf_path in pdf_list:
        pdf_result = parse_pdf(pdf_path, write_images = write_images)
        
        # Add batch processing metadata
        for page_data in pdf_result:
            page_data['batch_size'] = len(pdf_list)
            page_data['file_index'] = pdf_list.index(pdf_path)
        
        all_results.extend(pdf_result)
    
    return all_results


