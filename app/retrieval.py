# This file contains the functions for the text processing and document retrieval segment of the chatbot

import os
from typing import List, Dict, Any
import pymupdf4llm
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


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
                'page': page_metadata.get('page', 0),
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


def clean_text(text: str) -> str:
    """
    Clean text for better RAG performance while preserving markdown structure.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text optimized for embedding and chunking
    """
    if not text or not text.strip():
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Fix common PDF extraction artifacts
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove excessive whitespace while preserving structure
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\t+', ' ', text)  # Tabs to single space
    text = re.sub(r'\n +', '\n', text)  # Remove spaces after newlines
    text = re.sub(r' +\n', '\n', text)  # Remove spaces before newlines
    
    # Normalize line breaks (preserve paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'\r\n', '\n', text)  # Windows line endings to Unix
    text = re.sub(r'\r', '\n', text)  # Old Mac line endings to Unix
    
    # Clean up common PDF artifacts
    # Remove standalone page numbers (numbers on their own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove standalone roman numerals (common in headers/footers)
    text = re.sub(r'\n\s*[ivxlcdm]+\s*\n', '\n', text, flags = re.IGNORECASE)
    
    # Clean up markdown table formatting (preserve structure but clean spacing)
    # Fix spacing around table delimiters
    text = re.sub(r' +\| +', ' | ', text)  # Normalize spacing around pipes
    text = re.sub(r'^\| +', '| ', text, flags = re.MULTILINE)  # Start of line pipes
    text = re.sub(r' +\|$', ' |', text, flags = re.MULTILINE)  # End of line pipes
    
    # Preserve list formatting but clean spacing
    text = re.sub(r'\n +([•\-\*\+])', r'\n\1', text)  # Bullet lists
    text = re.sub(r'\n +(\d+\.)', r'\n\1', text)  # Numbered lists
    
    # Clean up header formatting (preserve markdown headers)
    text = re.sub(r'\n +(#+)', r'\n\1', text)  # Remove spaces before headers
    text = re.sub(r'(#+) +([^\n]+)', r'\1 \2', text)  # Normalize header spacing
    
    # Remove excessive punctuation (but preserve meaningful punctuation)
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'-{3,}', '---', text)  # Multiple dashes to em dash
    
    # Clean up quote marks
    text = re.sub(r'[\u201C\u201D\u201E]', '"', text)  # Normalize quotes
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize apostrophes
    
    # Remove zero-width characters and other invisible characters
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)
    
    # Final cleanup
    text = text.strip()  # Remove leading/trailing whitespace
    
    # Ensure text doesn't start or end with newlines after cleaning
    text = text.strip('\n')
    
    return text


def chunk_text_recursive(text: str, chunk_size: int = 500, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
    )
    
    # Split the text and return chunks
    chunks = text_splitter.split_text(text)
    
    return chunks


def access_chroma_collection(name: str):
    """
    Get or create a Chroma collection with the given name using ephemeral client.
    
    Args:
        name (str): Name of the collection
        
    Returns:
        Collection: ChromaDB collection object
    """
    client = chromadb.EphemeralClient()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = "BAAI/bge-small-en-v1.5"
    )
    collection = client.get_or_create_collection(name = name, embedding_function = sentence_transformer_ef)
    return collection



def preprocess_text(pages: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Clean and chunk text from parsed pages, retaining metadata.
    
    Args:
        pages (List[Dict[str, Any]]): Output from parse_pdf function
        chunk_size (int): Size for text chunking
        chunk_overlap (int): Overlap for text chunking
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries with metadata
    """
    chunk_documents = []
    
    for page in pages:
        # Clean the text
        cleaned_text = clean_text(page['text'])
        
        # Skip empty pages
        if not cleaned_text.strip():
            continue
            
        # Chunk the cleaned text
        chunks = chunk_text_recursive(cleaned_text, chunk_size, chunk_overlap)
        
        # Create chunk documents with metadata
        for chunk_num, chunk_text in enumerate(chunks):
            chunk_doc = {
                # Original page metadata
                'filename': page['filename'],
                'page': page['page'],
                'text_format': page['text_format'],
                'extraction_method': page['extraction_method'],
                'page_has_tables': page['has_tables'],
                'page_char_count': page['char_count'],
                'page_word_count': page['word_count'],
                'page_line_count': page['line_count'],
                'page_images_extracted': page['images_extracted'],
                'page_source_bbox': page['source_bbox'],
                'page_source_page_size': page['source_page_size'],
                # Chunk-specific data
                'text': chunk_text,
                'chunk_number': chunk_num + 1,
                'total_chunks_for_page': len(chunks),
                'chunk_char_count': len(chunk_text),
                'chunk_word_count': len(chunk_text.split()),
                'is_chunked': True,
                'chunk_size_used': chunk_size,
                'chunk_overlap_used': chunk_overlap
            }
            chunk_documents.append(chunk_doc)
    
    return chunk_documents


def add_documents(name: str, documents: List[Dict[str, Any]]) -> None:
    """
    Add documents to a ChromaDB collection.
    
    Args:
        name (str): Collection name
        documents (List[Dict[str, Any]]): List of document dictionaries
    """
    collection = access_chroma_collection(name)
    chunk_documents = preprocess_text(documents)

    # Prepare data for ChromaDB
    ids = []
    texts = []
    metadatas = []
    
    for doc in chunk_documents:
        # Create unique ID: {filename}_page{page}_chunk{chunk}
        doc_id = f"{doc['filename']}_page{doc['page']}_chunk{doc['chunk_number']}"
        ids.append(doc_id)
        texts.append(doc['text'])
        
        # Prepare metadata (exclude text and None values)
        metadata = {}
        for key, value in doc.items():
            if key != 'text' and value is not None:
                metadata[key] = value
        
        metadatas.append(metadata)
    
    # Add to collection
    collection.add(
        ids = ids,
        documents = texts,
        metadatas = metadatas
    )


def retrieve_documents(name: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query documents from a ChromaDB collection.
    
    Args:
        name (str): Collection name
        query (str): Query text
        top_k (int): Number of top results to return
        
    Returns:
        Dict[str, Any]: Query results from ChromaDB
    """
    collection = access_chroma_collection(name)
    
    results = collection.query(
        query_texts = [query],
        n_results = top_k
    )
    
    return results