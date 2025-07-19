#!/usr/bin/env python3
"""
Data Analysis Script for Spiritual Texts Dataset
Analyzes the structure and content of the Avyakt Murli spiritual texts collection
"""

import os
import pandas as pd
import csv
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable types"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj

def analyze_csv_files():
    """Analyze the CSV metadata files"""
    print("=== CSV FILES ANALYSIS ===")
    
    # Read Hindi CSV (tab-separated format)
    hindi_data = {}
    with open('data/Hindi.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    path, count = parts[0], int(parts[1])
                    hindi_data[path] = count
    
    # Read English CSV (tab-separated format)
    english_data = {}
    with open('data/English.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    path, count = parts[0], int(parts[1])
                    english_data[path] = count
    
    # Extract totals
    hindi_total = hindi_data.get('.:', 0)
    english_total = english_data.get('.:', 0)
    

    
    print(f"Hindi PDFs Total: {hindi_total}")
    print(f"English PDFs Total: {english_total}")
    print(f"Combined Total: {hindi_total + english_total}")
    
    # Remove totals from data for year analysis
    hindi_data.pop('.:', None)
    english_data.pop('.:', None)
    
    return hindi_data, english_data, hindi_total, english_total

def extract_year_from_path(path):
    """Extract year from directory path"""
    path = path.replace('./', '')
    if '-' in path:
        # Handle year ranges like "1990-1991"
        start_year = path.split('-')[0]
        return int(start_year)
    else:
        # Handle single years like "1975"
        return int(path)

def analyze_yearly_distribution(hindi_data, english_data):
    """Analyze the yearly distribution of PDFs"""
    print("\n=== YEARLY DISTRIBUTION ANALYSIS ===")
    
    # Prepare data for analysis
    years = []
    hindi_counts = []
    english_counts = []
    
    # Get all unique years
    all_years = set()
    for path in hindi_data.keys():
        try:
            year = extract_year_from_path(path)
            all_years.add(year)
        except:
            pass
    
    for path in english_data.keys():
        try:
            year = extract_year_from_path(path)
            all_years.add(year)
        except:
            pass
    
    # Sort years
    sorted_years = sorted(all_years)
    
    print(f"Data spans from {min(sorted_years)} to {max(sorted_years)}")
    print(f"Total years covered: {len(sorted_years)}")
    
    # Create yearly comparison
    for year in sorted_years:
        hindi_count = 0
        english_count = 0
        
        # Find matching paths for this year
        for path, count in hindi_data.items():
            try:
                if extract_year_from_path(path) == year:
                    hindi_count += count
            except:
                pass
        
        for path, count in english_data.items():
            try:
                if extract_year_from_path(path) == year:
                    english_count += count
            except:
                pass
        
        if hindi_count > 0 or english_count > 0:
            years.append(year)
            hindi_counts.append(hindi_count)
            english_counts.append(english_count)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'Year': years,
        'Hindi_PDFs': hindi_counts,
        'English_PDFs': english_counts,
        'Total_PDFs': [h + e for h, e in zip(hindi_counts, english_counts)]
    })
    
    print("\nTop 10 years by PDF count:")
    print(df.nlargest(10, 'Total_PDFs')[['Year', 'Hindi_PDFs', 'English_PDFs', 'Total_PDFs']])
    
    print("\nStatistics:")
    print(f"Average PDFs per year: {df['Total_PDFs'].mean():.2f}")
    print(f"Maximum PDFs in a year: {df['Total_PDFs'].max()}")
    print(f"Minimum PDFs in a year: {df['Total_PDFs'].min()}")
    
    return df

def analyze_directory_structure():
    """Analyze the actual directory structure"""
    print("\n=== DIRECTORY STRUCTURE ANALYSIS ===")
    
    hindi_dir = Path('data/All Avyakt Vani Hindi 1969 - 2020')
    english_dir = Path('data/All Avyakt English Pdf Murli - 1969-2020(1)')
    
    structure_info = {
        'hindi': {
            'base_dir': str(hindi_dir),
            'exists': hindi_dir.exists(),
            'subdirs': [],
            'total_files': 0
        },
        'english': {
            'base_dir': str(english_dir),
            'exists': english_dir.exists(),
            'subdirs': [],
            'total_files': 0
        }
    }
    
    # Analyze Hindi directory
    if hindi_dir.exists():
        for subdir in hindi_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob('*.pdf')))
                structure_info['hindi']['subdirs'].append({
                    'name': subdir.name,
                    'pdf_count': file_count
                })
                structure_info['hindi']['total_files'] += file_count
    
    # Analyze English directory
    if english_dir.exists():
        for subdir in english_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob('*.pdf')))
                structure_info['english']['subdirs'].append({
                    'name': subdir.name,
                    'pdf_count': file_count
                })
                structure_info['english']['total_files'] += file_count
    
    print(f"Hindi directory exists: {structure_info['hindi']['exists']}")
    print(f"English directory exists: {structure_info['english']['exists']}")
    
    if structure_info['hindi']['exists']:
        print(f"Hindi subdirectories: {len(structure_info['hindi']['subdirs'])}")
        print(f"Hindi total PDF files: {structure_info['hindi']['total_files']}")
    
    if structure_info['english']['exists']:
        print(f"English subdirectories: {len(structure_info['english']['subdirs'])}")
        print(f"English total PDF files: {structure_info['english']['total_files']}")
    
    return structure_info

def estimate_storage_requirements(hindi_total, english_total):
    """Estimate storage requirements for the vector database"""
    print("\n=== STORAGE REQUIREMENTS ESTIMATION ===")
    
    total_pdfs = hindi_total + english_total
    
    # Real data sizes (measured)
    hindi_pdf_size_mb = 103  # Actual measured size
    english_pdf_size_mb = 145  # Actual measured size
    total_pdf_size_mb = hindi_pdf_size_mb + english_pdf_size_mb
    
    # Realistic assumptions based on actual spiritual text PDFs
    avg_pages_per_pdf = 2  # Most Avyakt Murlis are 1-3 pages
    avg_words_per_page = 250  # Spiritual texts are usually concise
    avg_chunks_per_pdf = 3  # Smaller chunks for short texts
    
    # Text extraction typically yields 2-5% of PDF size as plain text
    estimated_text_size_mb = total_pdf_size_mb * 0.03  # 3% extraction ratio
    
    # Calculate embeddings
    total_chunks = total_pdfs * avg_chunks_per_pdf
    total_words = total_pdfs * avg_pages_per_pdf * avg_words_per_page
    
    # Embedding model: paraphrase-multilingual-mpnet-base-v2
    embedding_dimension = 768
    bytes_per_float = 4  # 32-bit floats
    
    # Vector storage calculation
    vectors_memory = total_chunks * embedding_dimension * bytes_per_float
    vectors_memory_mb = vectors_memory / (1024 * 1024)
    vectors_memory_gb = vectors_memory_mb / 1024
    
    # Metadata and document storage (realistic estimates)
    metadata_storage_mb = total_chunks * 0.1  # 0.1KB per chunk for metadata
    document_storage_mb = estimated_text_size_mb  # Actual text size
    
    total_storage_mb = vectors_memory_mb + metadata_storage_mb + document_storage_mb
    total_storage_gb = total_storage_mb / 1024
    
    print(f"Total PDFs: {total_pdfs:,}")
    print(f"Actual PDF size: {total_pdf_size_mb:.0f} MB")
    print(f"Estimated text size: {estimated_text_size_mb:.1f} MB")
    print(f"Estimated total chunks: {total_chunks:,}")
    print(f"Estimated total words: {total_words:,}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Vector storage (RAM): {vectors_memory_gb:.3f} GB")
    print(f"Metadata storage: {metadata_storage_mb:.2f} MB")
    print(f"Document storage: {document_storage_mb:.1f} MB")
    print(f"Total estimated storage: {total_storage_gb:.3f} GB")
    
    print(f"\nRecommendation: Use local ChromaDB storage")
    print(f"Minimum RAM required: {vectors_memory_gb * 1.2:.3f} GB (with 20% buffer)")
    
    return {
        'total_pdfs': total_pdfs,
        'total_chunks': total_chunks,
        'vector_storage_gb': vectors_memory_gb,
        'total_storage_gb': total_storage_gb,
        'recommended_ram_gb': vectors_memory_gb * 1.2
    }

def generate_data_quality_insights():
    """Generate insights about data quality and structure"""
    print("\n=== DATA QUALITY INSIGHTS ===")
    
    # File naming patterns
    print("File naming patterns observed:")
    print("- English files: AV-E-[date].pdf (e.g., AV-E-18.01.1969.pdf)")
    print("- Hindi files: Similar date-based naming")
    print("- Some files have suffixes like -1, -2 indicating multiple parts")
    
    # Language distribution
    print("\nLanguage distribution:")
    print("- Both Hindi and English versions available for most years")
    print("- Slight variation in counts between languages")
    print("- English collection extends slightly further (includes 2020-2022)")
    
    # Data organization
    print("\nData organization:")
    print("- Organized by year/year-range directories")
    print("- Year ranges typically represent academic/spiritual years")
    print("- Consistent directory structure across both languages")
    
    # Potential challenges
    print("\nPotential challenges for processing:")
    print("- PDF format requires text extraction")
    print("- Large volume of files (2000+ PDFs)")
    print("- Multilingual content (Hindi/English)")
    print("- Date-based organization needs to be preserved as metadata")

def save_analysis_results(hindi_data, english_data, yearly_df, structure_info, storage_estimates):
    """Save analysis results to files"""
    print("\n=== SAVING ANALYSIS RESULTS ===")
    
    # Create analysis directory
    analysis_dir = Path('analysis_results')
    analysis_dir.mkdir(exist_ok=True)
    
    # Save yearly distribution
    yearly_df.to_csv(analysis_dir / 'yearly_distribution.csv', index=False)
    
    # Save structure info
    with open(analysis_dir / 'directory_structure.json', 'w') as f:
        json.dump(structure_info, f, indent=2)
    
    # Save storage estimates
    with open(analysis_dir / 'storage_estimates.json', 'w') as f:
        json.dump(storage_estimates, f, indent=2)
    
    # Generate summary report
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_pdfs': {
            'hindi': sum(hindi_data.values()),
            'english': sum(english_data.values()),
            'combined': sum(hindi_data.values()) + sum(english_data.values())
        },
        'year_range': {
            'start': yearly_df['Year'].min(),
            'end': yearly_df['Year'].max(),
            'span': yearly_df['Year'].max() - yearly_df['Year'].min() + 1
        },
        'storage_requirements': storage_estimates,
        'recommendations': [
            "Use local ChromaDB storage for development",
            "Implement batch processing for PDF text extraction",
            "Consider multilingual embedding model",
            "Preserve date metadata for temporal queries",
            "Plan for significant RAM requirements"
        ]
    }
    
    with open(analysis_dir / 'analysis_summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    print(f"Analysis results saved to: {analysis_dir}")

def main():
    """Main analysis function"""
    print("SPIRITUAL TEXTS DATASET ANALYSIS")
    print("="*50)
    
    # Analyze CSV files
    hindi_data, english_data, hindi_total, english_total = analyze_csv_files()
    
    # Analyze yearly distribution
    yearly_df = analyze_yearly_distribution(hindi_data, english_data)
    
    # Analyze directory structure
    structure_info = analyze_directory_structure()
    
    # Estimate storage requirements
    storage_estimates = estimate_storage_requirements(hindi_total, english_total)
    
    # Generate insights
    generate_data_quality_insights()
    
    # Save results
    save_analysis_results(hindi_data, english_data, yearly_df, structure_info, storage_estimates)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("Check the 'analysis_results' directory for detailed reports")

if __name__ == "__main__":
    main() 