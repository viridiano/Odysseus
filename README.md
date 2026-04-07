# MultiModal FrameNet - Feature Extraction Pipeline

A GPU-accelerated pipeline for extracting structured semantic features from news photographs, built to support the expansion of FrameNet Brasil into the multimodal domain.

This project processes ~13,500 news images through a three-stage pipeline: multimodal LLM analysis, object detection with visual grounding, and semantic frame annotation. The output is a richly annotated dataset linking visual entities to FrameNet frames and frame elements in both English and Brazilian Portuguese.

## Pipeline Overview

```
Images (JPEG) → Gemma 3 12B → Structured CSV → GroundingDINO → Bounding Boxes → LOME/DAISY → Frame Annotations
                 (Stage 1)                       (Stage 2)                         (Stage 3)
```

**Stage 1 - Feature Extraction:** Each image is analyzed by Gemma 3 12B (4-bit quantized) to produce tagged entities, bilingual scene descriptions, event interpretations, and object lists.

**Stage 2 - Visual Grounding:** Entity lists are fed to GroundingDINO to generate bounding boxes with labels, grounding semantic descriptions to specific image regions.

**Stage 3 - Semantic Parsing:** Bilingual descriptions and grounded labels are processed through LOME + DAISY parsers by FrameNet Brasil to assign frames and frame elements.

## Setup

Tested on CWRU HPC with NVIDIA H100 and L40s GPUs, CUDA 12.1.

```bash
# Clone and setup
git clone https://github.com/nevernever69/MultiModal_Framenet.git
cd MultiModal_Framenet

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
./Download_dataset_extract.sh
```

For HPC usage, load the required modules first:
```bash
source config.sh
```

## Usage

### Stage 1: Feature Extraction with Gemma

```bash
# Single GPU
python Gemma_single_GPU.py

# Multi-GPU (auto-detects available GPUs)
python Gemma_parallel_GPU2.py

# Or submit as SLURM job
sbatch request_node.sh
```

Output: `structured_image_analysis_results.csv` with columns for entities, scene descriptions (EN/PT), event description, and object list.

### Stage 2: Object Detection with GroundingDINO

```bash
python grounding_dino_pipeline.py \
    --input_csv structured_image_analysis_results.csv \
    --output_csv grounding_dino_results.csv \
    --output_images_dir detected_images \
    --validate
```

Adds bounding box coordinates (both processed and original image dimensions) and saves labeled images.

### Post-Processing: Portuguese Entity Translation

```bash
python portuguese_entity_translation.py \
    --input_csv grounding_dino_results.csv \
    --output_csv final_results.csv \
    --model meta-llama/Llama-3.3-70B-Instruct
```

Extracts Portuguese entity names from existing Portuguese descriptions and adds them as a new column.

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Feature Extraction | `unsloth/gemma-3-12b-it-bnb-4bit` | Multimodal image analysis |
| Object Detection | `IDEA-Research/grounding-dino-base` | Bounding box generation |
| Entity Translation | `meta-llama/Llama-3.3-70B-Instruct` | Portuguese entity extraction |
| Semantic Parsing | LOME + DAISY | FrameNet frame assignment |

## Project Structure

```
├── Gemma_parallel_GPU2.py          # Main multi-GPU feature extraction (auto-detect GPUs)
├── Gemma_parallel_GPU.py           # 2-GPU parallel variant
├── Gemma_parallel_GPU1.py          # 2-GPU parallel variant (alt)
├── Gemma_single_GPU.py             # Single GPU version
├── gemma_analysis.py               # Original sequential script
├── grounding_dino_pipeline.py      # GroundingDINO object detection + coordinate fix
├── portuguese_entity_translation.py # Portuguese entity extraction via Llama
├── Download_dataset_extract.sh     # Dataset download
├── config.sh                       # HPC module setup
├── request_node.sh                 # SLURM batch job
├── interactive_node.sh             # Interactive GPU session
├── requirements.txt                # Python dependencies
└── PIPELINE_SUMMARY.md             # Detailed technical summary
```
