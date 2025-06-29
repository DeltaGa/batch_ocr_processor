# Advanced Batch OCR Processor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tesseract](https://img.shields.io/badge/tesseract-5.0+-green.svg)](https://github.com/tesseract-ocr/tesseract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art, enterprise-grade batch OCR processing system that leverages advanced Tesseract capabilities with **spatial text reconstruction** to preserve document layout, spacing, and formatting. Built with Python's most powerful libraries for high-performance, scalable document processing.

## 🌟 Key Features

### 🎯 **Spatial Text Reconstruction**
- **Layout Preservation**: Maintains original document spacing and formatting
- **Intelligent Word Positioning**: Uses bounding box analysis for accurate text reconstruction
- **Table & Column Support**: Handles structured documents with proper alignment
- **Multi-line Text**: Preserves paragraph breaks and text flow

### ⚡ **High-Performance Processing**
- **Multiple Execution Modes**: Sequential, Threading, Multiprocessing, and Async
- **Auto-scaling Workers**: Intelligent resource management based on system capabilities
- **Batch Processing**: Handle thousands of images efficiently
- **Progress Tracking**: Real-time processing status with detailed metrics

### 🔍 **Advanced Image Preprocessing**
- **PIL Enhancement Pipeline**: Contrast, sharpness, and brightness optimization
- **OpenCV Integration**: Advanced computer vision preprocessing
- **Noise Reduction**: Gaussian blur and morphological operations
- **Adaptive Thresholding**: Optimal binary conversion for text extraction

### 📊 **Comprehensive Output Formats**
- **JSON**: Structured data with full metadata and spatial information
- **CSV**: Tabular format for spreadsheet analysis
- **Excel**: Multi-sheet reports with summary statistics
- **Text Files**: Standard (with metadata) and clean (text-only) outputs
- **Individual Files**: Per-image text extraction with optional metadata
- **Combined Files**: All results in single files with configurable formatting
- **Word-Level Data**: Detailed bounding box coordinates for advanced analysis

### 🌍 **Multi-Language Support**
- Support for 100+ languages via Tesseract
- Configurable language models
- Mixed-language document processing
- Unicode text handling

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/batch-ocr-processor.git
cd batch-ocr-processor

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Create virtual environment
python -m venv ocr_env
source ocr_env/bin/activate  # Windows: ocr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process a single image
python batch_ocr.py document.jpg

# Process multiple images with layout preservation
python batch_ocr.py img1.jpg img2.png img3.tiff

# Process entire directory recursively
python batch_ocr.py /path/to/documents/ -r

# High-performance batch processing
python batch_ocr.py documents/ -m multiprocessing -w 8
```

## 📖 Advanced Usage Examples

### Spatial Layout Preservation
```bash
# Process forms with preserved spacing (default behavior)
python batch_ocr.py forms/ --preserve-layout

# Fine-tune spacing sensitivity for tables
python batch_ocr.py tables/ --space-threshold 40

# Export detailed word-level spatial data
python batch_ocr.py receipts/ --export-word-data
```

### Multi-Language Processing
```bash
# Spanish document processing
python batch_ocr.py documentos/ -l spa

# German documents with advanced preprocessing
python batch_ocr.py dokumente/ -l deu -p opencv

# Mixed language processing
python batch_ocr.py international/ -l eng+spa+fra
```

### Performance Optimization
```bash
# Maximum performance with all CPU cores
python batch_ocr.py large_batch/ -m multiprocessing -w $(nproc)

# Memory-optimized for large images
python batch_ocr.py high_res/ -m threading -w 4

# Async processing for I/O intensive workflows
python batch_ocr.py network_storage/ -m async
```

### Quality Control
```bash
# High-quality preprocessing for poor scans
python batch_ocr.py old_documents/ -p high

# OpenCV preprocessing for difficult images
python batch_ocr.py challenging/ -p opencv

# Verbose logging for debugging
python batch_ocr.py test_batch/ -v

# Quiet mode for automated processing
python batch_ocr.py images/ -q
```

### Quality Control
```bash
# High-quality preprocessing for poor scans
python batch_ocr.py old_documents/ -p high

# OpenCV preprocessing for difficult images
python batch_ocr.py challenging/ -p opencv

# Verbose logging for debugging
python batch_ocr.py test_batch/ -v
```

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  TesseractOCREngine │    │  ImagePreprocessor  │    │ BatchOCRProcessor   │
│                     │    │                     │    │                     │
│ • Spatial Recon.    │    │ • PIL Enhancement   │    │ • Multi-threading   │
│ • Confidence Calc.  │    │ • OpenCV Pipeline   │    │ • Multiprocessing   │
│ • Layout Analysis   │    │ • Noise Reduction   │    │ • Async Processing  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └─────────────┬─────────────┘                           │
                         │                                         │
                         ▼                                         ▼
              ┌─────────────────────┐                   ┌─────────────────────┐
              │    OCRResult        │                   │  OCRResultExporter  │
              │                     │                   │                     │
              │ • Text Content      │                   │ • JSON Export       │
              │ • Spatial Data      │                   │ • CSV Export        │
              │ • Confidence        │                   │ • Excel Export      │
              │ • Metadata          │                   │ • Text Files        │
              └─────────────────────┘                   └─────────────────────┘
```

### Spatial Text Reconstruction Algorithm

1. **Bounding Box Extraction**: Extract coordinates for every detected word
2. **Line Grouping**: Group words by vertical position using configurable thresholds
3. **Horizontal Analysis**: Calculate pixel gaps between adjacent words
4. **Smart Spacing**: Insert proportional spaces based on actual distances
5. **Layout Reconstruction**: Rebuild text maintaining original document structure

## 📊 Performance Benchmarks

| Image Type | Size | Processing Time | Throughput |
|------------|------|----------------|------------|
| Documents (300 DPI) | 1-2 MB | 1-3 seconds | ~1000 pages/hour |
| Forms | 0.5-1 MB | 0.5-2 seconds | ~1500 pages/hour |
| High-res Scans | 5-10 MB | 5-15 seconds | ~300 pages/hour |
| Mobile Photos | 2-5 MB | 2-8 seconds | ~600 pages/hour |

*Benchmarks on 8-core CPU with multiprocessing mode*

## 📁 Output Structure

```
ocr_output/
├── detailed_ocr_results.json         # Full results with spatial data
├── word_level_data.json              # Word coordinates and bounding boxes
├── ocr_results.csv                   # Tabular data for analysis
├── ocr_results.xlsx                  # Excel with multiple sheets
├── all_extracted_text.txt            # Combined text with metadata
├── all_extracted_text_clean.txt      # Combined text without metadata
├── clean_text_output.txt             # Text-only single file
├── extracted_texts/                  # Individual files with metadata
│   ├── document1.txt
│   ├── form2.txt
│   └── receipt3.txt
└── clean_texts/                      # Individual files without metadata
    ├── document1.txt
    ├── form2.txt
    └── receipt3.txt
```

### Text Output Formats

#### Standard Text Output (Default)
```
Source: /path/to/invoice.jpg
Confidence: 94.5%
Processing Time: 2.341s
--------------------------------------------------

INVOICE #12345
Date: 2024-01-15
Amount: $1,250.00
```

#### Clean Text Output (`--clean-text`)
```
INVOICE #12345
Date: 2024-01-15
Amount: $1,250.00
```

#### Text-Only Output (`--text-only`)
```
==================== invoice.jpg ====================

INVOICE #12345
Date: 2024-01-15
Amount: $1,250.00


==================== receipt.jpg ====================

RECEIPT
Store: ABC Market
Total: $45.67
```

## 🎯 Use Cases & Applications

### Document Processing Pipelines
- **Enterprise Document Management**: Process invoices, contracts, and forms with preserved layout
- **Data Extraction**: Clean text output for feeding into NLP models and databases
- **Archive Digitization**: Convert physical documents to searchable digital text
- **Compliance & Audit**: Maintain document integrity with comprehensive metadata

### Industry Applications
- **Financial Services**: Invoice processing, receipt digitization, form automation
- **Healthcare**: Medical record digitization, prescription processing
- **Legal**: Contract analysis, case document processing
- **Education**: Academic paper digitization, assignment processing
- **Manufacturing**: Quality control documentation, compliance records

### Integration Scenarios
- **API Workflows**: Clean text output for seamless API integration
- **Database Import**: Direct text content insertion without metadata cleanup
- **Content Management**: Pure text for CMS systems and search engines
- **Analytics Pipelines**: Structured data export for business intelligence tools

## 🔧 Configuration Options

### Command Line Arguments
| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `input_paths` | Files/directories to process | Required | `documents/` |
| `-r, --recursive` | Process directories recursively | False | `-r` |
| `-o, --output-dir` | Output directory | `ocr_output` | `-o results/` |
| `-f, --formats` | Output formats | `json` | `-f json csv xlsx` |
| `-l, --language` | OCR language | `eng` | `-l spa` |
| `-p, --preprocessing` | Image preprocessing level | `medium` | `-p opencv` |
| `-m, --mode` | Processing mode | `multiprocessing` | `-m threading` |
| `-w, --workers` | Number of workers | Auto-detect | `-w 8` |
| `--preserve-layout` | Enable spatial reconstruction | True | `--preserve-layout` |
| `--space-threshold` | Spacing sensitivity (pixels) | 30 | `--space-threshold 40` |
| `--export-word-data` | Export word-level coordinates | False | `--export-word-data` |
| `--individual` | Create individual text files | False | `--individual` |
| `-v, --verbose` | Verbose logging | False | `-v` |
| `-q, --quiet` | Suppress output | False | `-q` |

### Processing Modes

- **Sequential**: Single-threaded, minimal memory usage
- **Threading**: Multi-threaded, I/O bound optimization
- **Multiprocessing**: Multi-process, CPU intensive tasks
- **Async**: Asynchronous processing, modern concurrency

### Preprocessing Levels

- **None**: No preprocessing, fastest processing
- **Low**: Basic enhancement
- **Medium**: Balanced quality/speed (recommended)
- **High**: Aggressive enhancement for poor quality images
- **OpenCV**: Advanced computer vision preprocessing

## 🌍 Language Support

### Commonly Used Languages

| Language | Code | Installation |
|----------|------|--------------|
| English | `eng` | Default |
| Spanish | `spa` | `sudo apt install tesseract-ocr-spa` |
| French | `fra` | `sudo apt install tesseract-ocr-fra` |
| German | `deu` | `sudo apt install tesseract-ocr-deu` |
| Chinese (Simplified) | `chi-sim` | `sudo apt install tesseract-ocr-chi-sim` |
| Japanese | `jpn` | `sudo apt install tesseract-ocr-jpn` |
| Arabic | `ara` | `sudo apt install tesseract-ocr-ara` |

### Multi-Language Processing
```bash
# Process documents with multiple languages
python batch_ocr.py mixed_docs/ -l eng+spa+fra

# Auto-detect language (requires additional setup)
python batch_ocr.py unknown/ -l osd
```

## 🔍 Quality Assurance

### Confidence Scoring
- **Word-level confidence**: Individual word reliability scores
- **Weighted averaging**: Character-count based confidence calculation
- **Quality filtering**: Automatic low-confidence word filtering

### Error Handling
- **Graceful degradation**: Continue processing on individual failures
- **Detailed error logs**: Comprehensive error reporting
- **File integrity**: SHA256 hashing for verification

### Preprocessing Validation
- **Format verification**: Automatic image format detection
- **Dimension capture**: Image size and resolution tracking
- **Enhancement tracking**: Applied preprocessing steps logging

## 📈 Advanced Analytics

### Processing Statistics
```json
{
  "processing_stats": {
    "total_files": 150,
    "successful_extractions": 148,
    "failed_extractions": 2,
    "total_processing_time": 45.7,
    "average_confidence": 87.3,
    "layout_preserved_count": 146
  }
}
```

### Word-Level Analysis
```json
{
  "words": [
    {
      "text": "Invoice",
      "confidence": 96,
      "bbox": {"left": 100, "top": 50, "width": 120, "height": 25},
      "line_num": 1,
      "word_num": 1
    }
  ]
}
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY batch_ocr.py .
ENTRYPOINT ["python", "batch_ocr.py"]
```

### Kubernetes Scaling
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ocr-batch-job
spec:
  template:
    spec:
      containers:
      - name: ocr-processor
        image: batch-ocr:latest
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
          requests:
            cpu: "2"
            memory: "4Gi"
```

### Monitoring & Alerting
- **Processing rate monitoring**: Track documents per hour
- **Error rate alerts**: Automated failure notifications
- **Resource utilization**: CPU and memory tracking
- **Quality metrics**: Confidence score trends

## 🔧 Troubleshooting

### Common Issues

#### "tesseract is not installed or not in PATH"
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: Add to PATH after installation
```

#### Poor OCR Quality
```bash
# Try different preprocessing
python batch_ocr.py images/ -p opencv

# Use appropriate language
python batch_ocr.py docs/ -l spa

# Combine both approaches
python batch_ocr.py docs/ -l deu -p high
```

#### Memory Issues
```bash
# Reduce workers
python batch_ocr.py large_batch/ -w 2

# Use sequential processing
python batch_ocr.py huge_files/ -m sequential
```

#### Layout Issues with Clean Text
```bash
# Adjust spacing threshold for better layout preservation
python batch_ocr.py forms/ --space-threshold 50 --clean-text

# Export word data for manual layout analysis
python batch_ocr.py tables/ --export-word-data --text-only

# Use different output formats for comparison
python batch_ocr.py docs/ --individual --combined --clean-text
```

### Performance Optimization

1. **Image Quality**: Higher DPI generally improves accuracy
2. **Preprocessing**: Match preprocessing level to image quality
3. **Language Models**: Use specific language models when possible
4. **Worker Count**: Optimal workers = CPU cores + 4 for I/O bound tasks
5. **Batch Size**: Process 100-1000 images per batch for optimal performance

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **CPU**: 2 cores
- **Storage**: 1GB free space
- **Tesseract**: 4.0+

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 8GB+
- **CPU**: 4+ cores
- **Storage**: SSD with 5GB+ free space
- **Tesseract**: 5.0+

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-repo/batch-ocr-processor.git
cd batch-ocr-processor
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
python -m pytest --cov=batch_ocr tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for the powerful OCR engine
- [OpenCV](https://opencv.org/) for advanced image processing capabilities
- [Pillow](https://python-pillow.org/) for image manipulation
- [PyTesseract](https://github.com/madmaze/pytesseract) for Python integration

## 📞 Support

- **Documentation**: [Full documentation](https://your-docs-site.com)
- **Issues**: [GitHub Issues](https://github.com/your-repo/batch-ocr-processor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/batch-ocr-processor/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for the OCR community**

*Transform your document processing workflows with enterprise-grade OCR capabilities.*
