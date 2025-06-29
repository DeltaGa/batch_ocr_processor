#!/usr/bin/env python3
"""
Advanced Batch OCR Processing System
====================================

A high-performance, enterprise-grade OCR processing system that leverages
Tesseract for text extraction with comprehensive error handling, parallel
processing, and flexible output formatting.

Author: Advanced Python Development
Version: 2.1.0
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import mimetypes
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import hashlib

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Configure logging with advanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats for OCR results."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    XML = "xml"
    XLSX = "xlsx"


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNC = "async"


@dataclass
class OCRResult:
    """Structured container for OCR processing results."""
    file_path: str
    file_name: str
    file_size: int
    processing_time: float
    confidence_score: float
    text_content: str
    word_count: int
    character_count: int
    language: str
    timestamp: str
    file_hash: str
    error_message: Optional[str] = None
    preprocessing_applied: List[str] = field(default_factory=list)
    tesseract_version: Optional[str] = None
    image_dimensions: Optional[Tuple[int, int]] = None
    spatial_data: Optional[Dict] = None
    layout_preserved: bool = False
    processing_method: str = "spatial_reconstruction"

    def to_dict(self) -> Dict:
        """Convert OCRResult to dictionary format."""
        result_dict = asdict(self)
        # Convert spatial_data to string representation for CSV compatibility
        if result_dict.get('spatial_data'):
            result_dict['spatial_data'] = str(result_dict['spatial_data'])
        return result_dict


class ImagePreprocessor:
    """Advanced image preprocessing for enhanced OCR accuracy."""
    
    @staticmethod
    def enhance_image(image: Image.Image, enhancement_level: str = "medium") -> Image.Image:
        """Apply comprehensive image enhancement techniques."""
        enhanced = image.copy()
        
        if enhancement_level in ["medium", "high"]:
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
        
        if enhancement_level == "high":
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # Apply denoising filter
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        return enhanced
    
    @staticmethod
    def opencv_preprocessing(image_path: str) -> np.ndarray:
        """Advanced OpenCV-based preprocessing pipeline."""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed


class TesseractOCREngine:
    """High-performance Tesseract OCR wrapper with spatial text reconstruction."""
    
    def __init__(self, language: str = "eng", oem: int = 3, psm: int = 6):
        """
        Initialize OCR engine with optimized parameters.
        
        Args:
            language: Tesseract language code
            oem: OCR Engine Mode (0-3)
            psm: Page Segmentation Mode (0-13)
        """
        self.language = language
        self.oem = oem
        self.psm = psm
        self.tesseract_version = self._get_tesseract_version()
        
        # Optimized Tesseract configuration for better text extraction
        self.config = f'--oem {oem} --psm {psm}'
    
    def _get_tesseract_version(self) -> str:
        """Retrieve Tesseract version information."""
        try:
            return pytesseract.get_tesseract_version()
        except Exception:
            return "Unknown"
    
    def _reconstruct_text_with_spacing(self, data: Dict) -> str:
        """
        Reconstruct text with proper spacing using bounding box analysis.
        
        Args:
            data: Tesseract OCR data dictionary
            
        Returns:
            Properly formatted text with preserved spacing
        """
        if not data['text']:
            return ""
        
        # Group words by line using vertical position
        lines = {}
        word_threshold = 10  # Pixels threshold for same line detection
        space_threshold = 30  # Pixels threshold for space insertion
        
        for i, text in enumerate(data['text']):
            if not text.strip():  # Skip empty text
                continue
                
            conf = int(data['conf'][i])
            if conf <= 0:  # Skip low confidence detections
                continue
            
            # Get bounding box coordinates
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            # Find the appropriate line based on vertical position
            line_key = None
            for existing_y in lines.keys():
                if abs(y - existing_y) <= word_threshold:
                    line_key = existing_y
                    break
            
            if line_key is None:
                line_key = y
                lines[line_key] = []
            
            lines[line_key].append({
                'text': text.strip(),
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'conf': conf,
                'right': x + w
            })
        
        # Sort lines by vertical position (top to bottom)
        sorted_lines = sorted(lines.items())
        
        reconstructed_text = []
        
        for line_y, words in sorted_lines:
            if not words:
                continue
                
            # Sort words by horizontal position (left to right)
            words.sort(key=lambda w: w['x'])
            
            line_text = []
            prev_right = 0
            
            for word in words:
                # Calculate spacing based on gap between words
                if prev_right > 0:
                    gap = word['x'] - prev_right
                    if gap > space_threshold:
                        # Large gap - add multiple spaces or tab
                        if gap > space_threshold * 3:
                            line_text.append('\t')  # Very large gap = tab
                        else:
                            spaces_needed = max(1, gap // 10)  # Proportional spacing
                            line_text.append(' ' * min(spaces_needed, 10))  # Cap at 10 spaces
                    else:
                        line_text.append(' ')  # Normal space
                
                line_text.append(word['text'])
                prev_right = word['right']
            
            if line_text:
                reconstructed_text.append(''.join(line_text))
        
        return '\n'.join(reconstructed_text)
    
    def _calculate_confidence_score(self, data: Dict) -> float:
        """
        Calculate weighted confidence score based on text length and confidence.
        
        Args:
            data: Tesseract OCR data dictionary
            
        Returns:
            Weighted average confidence score
        """
        total_conf = 0
        total_chars = 0
        
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
                
            conf = int(data['conf'][i])
            if conf > 0:
                char_count = len(text.strip())
                total_conf += conf * char_count
                total_chars += char_count
        
        return total_conf / total_chars if total_chars > 0 else 0.0
    
    def extract_text_with_confidence(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
        """
        Extract text with proper spacing preservation and confidence scoring.
        
        Returns:
            Tuple of (extracted_text_with_spacing, confidence_score)
        """
        try:
            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(
                image, 
                lang=self.language, 
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Reconstruct text with proper spacing
            reconstructed_text = self._reconstruct_text_with_spacing(data)
            
            # Calculate weighted confidence score
            confidence_score = self._calculate_confidence_score(data)
            
            return reconstructed_text, confidence_score
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return "", 0.0
    
    def extract_detailed_data(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """
        Extract comprehensive OCR data including word-level information.
        
        Returns:
            Dictionary containing detailed OCR analysis
        """
        try:
            # Get word-level data
            data = pytesseract.image_to_data(
                image, 
                lang=self.language, 
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process and structure the data
            words_data = []
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                    
                conf = int(data['conf'][i])
                if conf > 0:
                    words_data.append({
                        'text': text.strip(),
                        'confidence': conf,
                        'bbox': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'level': data['level'][i],
                        'page_num': data['page_num'][i],
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    })
            
            return {
                'words': words_data,
                'reconstructed_text': self._reconstruct_text_with_spacing(data),
                'average_confidence': self._calculate_confidence_score(data),
                'total_words': len(words_data),
                'processing_method': 'spatial_reconstruction'
            }
            
        except Exception as e:
            logger.error(f"Detailed OCR extraction failed: {str(e)}")
            return {
                'words': [],
                'reconstructed_text': "",
                'average_confidence': 0.0,
                'total_words': 0,
                'processing_method': 'failed',
                'error': str(e)
            }


class BatchOCRProcessor:
    """Enterprise-grade batch OCR processing system."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    def __init__(self, 
                 language: str = "eng",
                 preprocessing: str = "medium",
                 processing_mode: ProcessingMode = ProcessingMode.MULTIPROCESSING,
                 max_workers: int = None):
        """
        Initialize the batch OCR processor.
        
        Args:
            language: OCR language code
            preprocessing: Image preprocessing level
            processing_mode: Execution mode for parallel processing
            max_workers: Maximum number of worker processes/threads
        """
        self.ocr_engine = TesseractOCREngine(language=language)
        self.preprocessor = ImagePreprocessor()
        self.preprocessing_level = preprocessing
        self.processing_mode = processing_mode
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
        self.results: List[OCRResult] = []
        self.processing_stats = {
            'total_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_processing_time': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for integrity verification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return "unknown"
    
    def _is_supported_image(self, file_path: Path) -> bool:
        """Validate if file is a supported image format."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def _discover_images(self, input_paths: List[str], recursive: bool = False) -> List[Path]:
        """
        Discover all supported image files from input paths.
        
        Args:
            input_paths: List of file or directory paths
            recursive: Whether to search directories recursively
            
        Returns:
            List of valid image file paths
        """
        image_files = []
        
        for path_str in input_paths:
            path = Path(path_str)
            
            if path.is_file() and self._is_supported_image(path):
                image_files.append(path)
            elif path.is_dir():
                pattern = "**/*" if recursive else "*"
                for file_path in path.glob(pattern):
                    if file_path.is_file() and self._is_supported_image(file_path):
                        image_files.append(file_path)
        
        return sorted(set(image_files))  # Remove duplicates and sort
    
    def _process_single_image(self, image_path: Path) -> OCRResult:
        """
        Process a single image file with comprehensive error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCRResult object containing processing results
        """
        start_time = time.time()
        file_size = image_path.stat().st_size
        file_hash = self._calculate_file_hash(str(image_path))
        preprocessing_applied = []
        
        try:
            # Load and preprocess image
            if self.preprocessing_level == "opencv":
                processed_image = self.preprocessor.opencv_preprocessing(str(image_path))
                preprocessing_applied.append("opencv_pipeline")
            else:
                image = Image.open(image_path)
                if self.preprocessing_level != "none":
                    image = self.preprocessor.enhance_image(image, self.preprocessing_level)
                    preprocessing_applied.append(f"pil_enhancement_{self.preprocessing_level}")
                processed_image = image
            
            # Get image dimensions
            if hasattr(processed_image, 'shape'):  # numpy array
                dimensions = (processed_image.shape[1], processed_image.shape[0])
            else:  # PIL Image
                dimensions = processed_image.size
            
            # Perform OCR extraction with spatial reconstruction
            detailed_data = self.ocr_engine.extract_detailed_data(processed_image)
            text_content = detailed_data['reconstructed_text']
            confidence_score = detailed_data['average_confidence']
            
            processing_time = time.time() - start_time
            
            # Create result object
            result = OCRResult(
                file_path=str(image_path),
                file_name=image_path.name,
                file_size=file_size,
                processing_time=processing_time,
                confidence_score=confidence_score,
                text_content=text_content,
                word_count=len(text_content.split()) if text_content else 0,
                character_count=len(text_content),
                language=self.ocr_engine.language,
                timestamp=datetime.now().isoformat(),
                file_hash=file_hash,
                preprocessing_applied=preprocessing_applied,
                tesseract_version=str(self.ocr_engine.tesseract_version),
                image_dimensions=dimensions,
                spatial_data=detailed_data,
                layout_preserved=True,
                processing_method=detailed_data.get('processing_method', 'spatial_reconstruction')
            )
            
            logger.info(f"Successfully processed: {image_path.name} ({confidence_score:.1f}% confidence)")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Failed to process {image_path.name}: {error_msg}")
            
            return OCRResult(
                file_path=str(image_path),
                file_name=image_path.name,
                file_size=file_size,
                processing_time=processing_time,
                confidence_score=0.0,
                text_content="",
                word_count=0,
                character_count=0,
                language=self.ocr_engine.language,
                timestamp=datetime.now().isoformat(),
                file_hash=file_hash,
                error_message=error_msg,
                preprocessing_applied=preprocessing_applied,
                tesseract_version=str(self.ocr_engine.tesseract_version)
            )
    
    def process_batch(self, input_paths: List[str], recursive: bool = False) -> List[OCRResult]:
        """
        Process multiple images using the configured processing mode.
        
        Args:
            input_paths: List of file or directory paths
            recursive: Whether to search directories recursively
            
        Returns:
            List of OCRResult objects
        """
        self.processing_stats['start_time'] = datetime.now()
        
        # Discover all image files
        image_files = self._discover_images(input_paths, recursive)
        self.processing_stats['total_files'] = len(image_files)
        
        if not image_files:
            logger.warning("No supported image files found in the specified paths")
            return []
        
        logger.info(f"Found {len(image_files)} image files to process")
        logger.info(f"Using {self.processing_mode.value} processing mode with {self.max_workers} workers")
        
        # Process images based on selected mode
        if self.processing_mode == ProcessingMode.SEQUENTIAL:
            results = self._process_sequential(image_files)
        elif self.processing_mode == ProcessingMode.THREADING:
            results = self._process_with_threading(image_files)
        elif self.processing_mode == ProcessingMode.MULTIPROCESSING:
            results = self._process_with_multiprocessing(image_files)
        else:  # ASYNC
            results = asyncio.run(self._process_async(image_files))
        
        # Update processing statistics
        self.processing_stats['end_time'] = datetime.now()
        self.processing_stats['successful_extractions'] = sum(1 for r in results if not r.error_message)
        self.processing_stats['failed_extractions'] = sum(1 for r in results if r.error_message)
        self.processing_stats['total_processing_time'] = sum(r.processing_time for r in results)
        
        self.results = results
        return results
    
    def _process_sequential(self, image_files: List[Path]) -> List[OCRResult]:
        """Process images sequentially with progress bar."""
        results = []
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for image_path in image_files:
                result = self._process_single_image(image_path)
                results.append(result)
                pbar.update(1)
        return results
    
    def _process_with_threading(self, image_files: List[Path]) -> List[OCRResult]:
        """Process images using thread pool executor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(self._process_single_image, path): path 
                            for path in image_files}
            
            with tqdm(total=len(image_files), desc="Processing images (threaded)") as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        return results
    
    def _process_with_multiprocessing(self, image_files: List[Path]) -> List[OCRResult]:
        """Process images using process pool executor."""
        # Note: For multiprocessing, we need to pass the processor configuration
        # This is a simplified version - in production, you'd want to handle
        # the serialization of the processor state properly
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(self._process_single_image, path): path 
                            for path in image_files}
            
            with tqdm(total=len(image_files), desc="Processing images (multiprocess)") as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        return results
    
    async def _process_async(self, image_files: List[Path]) -> List[OCRResult]:
        """Process images using async/await pattern."""
        loop = asyncio.get_event_loop()
        
        # Create tasks for concurrent execution
        tasks = []
        for image_path in image_files:
            task = loop.run_in_executor(None, self._process_single_image, image_path)
            tasks.append(task)
        
        # Execute with progress tracking
        results = []
        with tqdm(total=len(tasks), desc="Processing images (async)") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        return results


class OCRResultExporter:
    """Advanced export system for OCR results with multiple format support."""
    
    def __init__(self, results: List[OCRResult], output_dir: str = "ocr_output"):
        """
        Initialize the exporter with OCR results.
        
        Args:
            results: List of OCRResult objects
            output_dir: Output directory for generated files
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def export_detailed_json(self, filename: str = "detailed_ocr_results.json") -> Path:
        """Export results with full spatial data to JSON format."""
        output_path = self.output_dir / filename
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(self.results),
                'successful_extractions': sum(1 for r in self.results if not r.error_message),
                'failed_extractions': sum(1 for r in self.results if r.error_message),
                'layout_preserved_count': sum(1 for r in self.results if r.layout_preserved),
                'processing_methods': list(set(r.processing_method for r in self.results))
            },
            'results': []
        }
        
        for result in self.results:
            result_data = result.to_dict()
            # Keep spatial_data as proper dict for JSON
            if result.spatial_data:
                result_data['spatial_data'] = result.spatial_data
            export_data['results'].append(result_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed JSON export completed: {output_path}")
        return output_path
    
    def export_word_level_data(self, filename: str = "word_level_data.json") -> Path:
        """Export word-level OCR data with bounding boxes."""
        output_path = self.output_dir / filename
        
        word_level_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'description': 'Word-level OCR data with spatial coordinates'
            },
            'files': []
        }
        
        for result in self.results:
            if result.spatial_data and 'words' in result.spatial_data:
                file_data = {
                    'file_path': result.file_path,
                    'file_name': result.file_name,
                    'confidence_score': result.confidence_score,
                    'word_count': len(result.spatial_data['words']),
                    'words': result.spatial_data['words']
                }
                word_level_data['files'].append(file_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(word_level_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Word-level data export completed: {output_path}")
        return output_path
    
    def export_csv(self, filename: str = "ocr_results.csv") -> Path:
        """Export results to CSV format."""
        output_path = self.output_dir / filename
        
        # Convert results to DataFrame
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        # Handle list columns by converting to string
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"CSV export completed: {output_path}")
        return output_path
    
    def export_excel(self, filename: str = "ocr_results.xlsx") -> Path:
        """Export results to Excel format with multiple sheets."""
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results sheet
            df = pd.DataFrame([result.to_dict() for result in self.results])
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
            
            df.to_sheet(writer, sheet_name='OCR_Results', index=False)
            
            # Summary statistics sheet
            summary_data = {
                'Metric': ['Total Files', 'Successful Extractions', 'Failed Extractions', 
                          'Average Confidence', 'Total Processing Time (s)', 'Average Processing Time (s)'],
                'Value': [
                    len(self.results),
                    sum(1 for r in self.results if not r.error_message),
                    sum(1 for r in self.results if r.error_message),
                    np.mean([r.confidence_score for r in self.results if r.confidence_score > 0]),
                    sum(r.processing_time for r in self.results),
                    np.mean([r.processing_time for r in self.results])
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel export completed: {output_path}")
        return output_path
    
    def export_text_files(self, individual_files: bool = True, combined_file: bool = True, clean_output: bool = False) -> List[Path]:
        """Export extracted text to individual and/or combined text files."""
        output_paths = []
        
        if individual_files:
            text_dir = self.output_dir / ("clean_texts" if clean_output else "extracted_texts")
            text_dir.mkdir(exist_ok=True)
            
            for result in self.results:
                if result.text_content and not result.error_message:
                    safe_filename = "".join(c for c in result.file_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                    text_filename = f"{safe_filename}.txt"
                    text_path = text_dir / text_filename
                    
                    with open(text_path, 'w', encoding='utf-8') as f:
                        if clean_output:
                            # Clean output: only the extracted text
                            f.write(result.text_content)
                        else:
                            # Standard output: with metadata
                            f.write(f"Source: {result.file_path}\n")
                            f.write(f"Confidence: {result.confidence_score:.2f}%\n")
                            f.write(f"Processing Time: {result.processing_time:.3f}s\n")
                            f.write("-" * 50 + "\n\n")
                            f.write(result.text_content)
                    
                    output_paths.append(text_path)
        
        if combined_file:
            filename = "all_extracted_text_clean.txt" if clean_output else "all_extracted_text.txt"
            combined_path = self.output_dir / filename
            
            with open(combined_path, 'w', encoding='utf-8') as f:
                if clean_output:
                    # Clean combined output: only text content separated by double newlines
                    text_contents = []
                    for result in self.results:
                        if result.text_content and not result.error_message:
                            text_contents.append(result.text_content.strip())
                    
                    f.write("\n\n".join(text_contents))
                else:
                    # Standard combined output: with full metadata
                    f.write(f"Combined OCR Results - Generated on {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for result in self.results:
                        f.write(f"File: {result.file_name}\n")
                        f.write(f"Path: {result.file_path}\n")
                        f.write(f"Confidence: {result.confidence_score:.2f}%\n")
                        f.write(f"Word Count: {result.word_count}\n")
                        f.write(f"Processing Time: {result.processing_time:.3f}s\n")
                        
                        if result.error_message:
                            f.write(f"Error: {result.error_message}\n")
                        else:
                            f.write("-" * 40 + "\n")
                            f.write(result.text_content)
                            f.write("\n")
                        
                        f.write("\n" + "=" * 80 + "\n\n")
            
            output_paths.append(combined_path)
            logger.info(f"{'Clean' if clean_output else 'Standard'} combined text export completed: {combined_path}")
        
        return output_paths
    
    def export_clean_text_only(self, filename: str = "clean_text_output.txt") -> Path:
        """Export only the extracted text content without any metadata or formatting."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(self.results):
                if result.text_content and not result.error_message:
                    # Add separator between files (except for the first one)
                    if i > 0:
                        f.write("\n" + "="*20 + f" {result.file_name} " + "="*20 + "\n\n")
                    else:
                        f.write(f"="*20 + f" {result.file_name} " + "="*20 + "\n\n")
                    
                    f.write(result.text_content.strip())
                    f.write("\n\n")
        
        logger.info(f"Clean text-only export completed: {output_path}")
        return output_path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Batch OCR Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                          # Process single image
  %(prog)s images/ -r                         # Process directory recursively
  %(prog)s img1.jpg img2.png -o results/      # Process multiple files
  %(prog)s images/ -f json csv -l spa         # Spanish OCR, JSON+CSV output
  %(prog)s images/ -m multiprocessing -w 8    # 8-process parallel processing
  %(prog)s images/ -p opencv --individual     # OpenCV preprocessing, individual text files
  %(prog)s forms/ --clean-text --individual   # Clean text output without metadata
  %(prog)s docs/ --text-only                  # Single file with only extracted text
  %(prog)s receipts/ --combined --clean-text  # Clean combined text file
        """
    )
    
    # Input arguments
    parser.add_argument(
        'input_paths',
        nargs='+',
        help='Input image files or directories to process'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    
    # Output arguments
    parser.add_argument(
        '-o', '--output-dir',
        default='ocr_output',
        help='Output directory for results (default: ocr_output)'
    )
    
    parser.add_argument(
        '-f', '--formats',
        nargs='+',
        choices=[fmt.value for fmt in OutputFormat],
        default=['json'],
        help='Output formats (default: json)'
    )
    
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Create individual text files for each processed image'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        default=True,
        help='Create combined text file with all results (default: True)'
    )
    
    parser.add_argument(
        '--clean-text',
        action='store_true',
        help='Export clean text output without metadata headers'
    )
    
    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Export a single file with only extracted text content (no metadata)'
    )
    
    # OCR configuration
    parser.add_argument(
        '-l', '--language',
        default='eng',
        help='Tesseract language code (default: eng)'
    )
    
    parser.add_argument(
        '-p', '--preprocessing',
        choices=['none', 'low', 'medium', 'high', 'opencv'],
        default='medium',
        help='Image preprocessing level (default: medium)'
    )
    
    parser.add_argument(
        '--preserve-layout',
        action='store_true',
        default=True,
        help='Preserve text layout and spacing using spatial reconstruction (default: True)'
    )
    
    parser.add_argument(
        '--space-threshold',
        type=int,
        default=30,
        help='Pixel threshold for space insertion between words (default: 30)'
    )
    
    parser.add_argument(
        '--export-word-data',
        action='store_true',
        help='Export detailed word-level data with bounding boxes'
    )
    
    # Processing configuration
    parser.add_argument(
        '-m', '--mode',
        choices=[mode.value for mode in ProcessingMode],
        default='multiprocessing',
        help='Processing mode (default: multiprocessing)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        help='Number of worker processes/threads (default: auto-detect)'
    )
    
    # Logging and output control
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser


def main():
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Initialize OCR processor
        processor = BatchOCRProcessor(
            language=args.language,
            preprocessing=args.preprocessing,
            processing_mode=ProcessingMode(args.mode),
            max_workers=args.workers
        )
        
        # Process images
        logger.info("Starting batch OCR processing...")
        results = processor.process_batch(args.input_paths, args.recursive)
        
        if not results:
            logger.error("No images were processed successfully")
            sys.exit(1)
        
        # Export results
        exporter = OCRResultExporter(results, args.output_dir)
        
        output_files = []
        
        for format_name in args.formats:
            output_format = OutputFormat(format_name)
            
            if output_format == OutputFormat.JSON:
                output_files.append(exporter.export_detailed_json())
            elif output_format == OutputFormat.CSV:
                output_files.append(exporter.export_csv())
            elif output_format == OutputFormat.XLSX:
                output_files.append(exporter.export_excel())
        
        # Always export word-level data for advanced analysis
        if any(r.spatial_data for r in results):
            output_files.append(exporter.export_word_level_data())
        
        # Export text files if requested
        if args.individual or args.combined:
            text_files = exporter.export_text_files(
                individual_files=args.individual, 
                combined_file=args.combined,
                clean_output=args.clean_text
            )
            output_files.extend(text_files)
        
        # Export clean text-only file if requested
        if args.text_only:
            clean_text_file = exporter.export_clean_text_only()
            output_files.append(clean_text_file)
        
        # Print processing summary
        stats = processor.processing_stats
        logger.info("Processing completed successfully!")
        logger.info(f"Files processed: {stats['total_files']}")
        logger.info(f"Successful extractions: {stats['successful_extractions']}")
        logger.info(f"Failed extractions: {stats['failed_extractions']}")
        logger.info(f"Total processing time: {stats['total_processing_time']:.2f}s")
        logger.info(f"Output files created: {len(output_files)}")
        
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()