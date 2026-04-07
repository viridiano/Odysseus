## Technical Summary: Feature Extraction Pipeline for Multimodal FrameNet Expansion

### High-Level Overview

This pipeline extracts structured semantic features from ~13,500 news photographs to support the expansion of FrameNet into the multimodal domain. It transforms raw images into richly annotated records containing tagged entities, bilingual scene descriptions, event interpretations, and visually grounded object lists -- which are then passed through object detection and semantic parsing to produce frame-annotated, bounding-box-level data.

**Inputs:** JPEG news photographs from a curated dataset of 13,000+ images.

**Outputs (per image):**
- Entities & Relationships (semantically tagged)
- Scene Description in English
- Scene Description in Brazilian Portuguese
- Event Description
- Entity/Object List
- Bounding box coordinates with labels (from GroundingDINO)
- FrameNet frame and frame element assignments (from LOME + DAISY)

### Pipeline Stages

**Stage 1 -- LLM-Based Feature Extraction**

Each image is analyzed by **Gemma 3 12B** (instruction-tuned, 4-bit quantized via Unsloth) using a carefully designed multi-section prompt. The model produces structured markdown output covering all five annotation fields in a single inference call. The bilingual design -- generating both English and Portuguese descriptions in one pass -- keeps entity tags consistent across languages and reduces compute cost.

The prompt uses a controlled entity taxonomy of 20+ types (`[person]`, `[clothing]`, `[vehicle]`, `[text]`, etc.) and instructs the model to tag every mentioned entity inline. Scene descriptions emphasize spatial relationships and observable context while avoiding identity inference. Event descriptions allow up to three plausible interpretations for ambiguous scenes.

Output is parsed via regex to extract each section, cleaned of markdown artifacts, and exported to CSV.

**Stage 2 -- Visual Grounding with GroundingDINO**

The entity/object lists from Stage 1 are fed as text prompts to **GroundingDINO** (`IDEA-Research/grounding-dino-base`), which localizes each entity in the image with bounding boxes and confidence scores (`box_threshold=0.3`, `text_threshold=0.25`). This grounds the LLM's semantic descriptions to specific image regions, producing `Detected_Boxes_Coordinates` and `Detected_Boxes_Image` columns. A coordinate conversion step maps GroundingDINO's internally upscaled coordinates back to original image dimensions (`Original_Image_Coordinates`), since the processor resizes images to varying resolutions with no consistent scaling factor. Coordinate validation and visual verification scripts were used to ensure bounding box accuracy.

**Stage 3 -- Semantic Frame Annotation**

The bilingual descriptions and grounded entity labels are processed through **LOME** and **DAISY** semantic parsers by the FrameNet Brasil team. Each bounding box receives a triplet: Frame label, Frame Element label, and CV Name -- linking visual entities to FrameNet's semantic inventory. The Portuguese descriptions provide crucial contextual alignment for the Portuguese-language frame inventory.

### Infrastructure & Performance

All GPU-intensive stages (feature extraction, GroundingDINO) were run on the **CWRU HPC cluster** (SLURM) using NVIDIA H100 and L40s GPUs with CUDA 12.1. The feature extraction pipeline uses Python's `multiprocessing.Pool` with round-robin GPU assignment -- each GPU runs its own model instance processing different images concurrently, achieving true parallel throughput. With 3 GPUs, the full dataset was processed in approximately 60 hours (~40 seconds per image depending on complexity).

4-bit quantization via bitsandbytes enabled fitting the 12B-parameter model on available hardware. `torch.compile()` and `bfloat16` precision were applied for additional optimization.

### Post-Processing

- **Portuguese entity translations** were added using **Llama 3.3 70B**, extracting Portuguese entity names from the existing translated descriptions.
- A **consistency flag** column identifies where bounding box labels diverge from scene description wording: 53.4% of rows (7,231) were fully consistent; the remainder were flagged for human validation, with most mismatches being semantically equivalent variations (e.g., "glue jar" vs. "jar of glue").

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Gemma 3 12B over BLIP/LLaVA | Earlier models were tested but produced less reliable structured outputs. Gemma 3 followed the multi-section prompt format consistently. |
| 4-bit quantization (Unsloth) | Enabled fitting the 12B model on available GPUs with minimal quality loss -- critical for processing 13K+ images at scale. |
| Round-robin multi-GPU over `device_map` | `device_map` splits one model across GPUs (same throughput as single GPU). Round-robin assigns different images to separate GPU-local instances, achieving true parallelism. |
| Bilingual output in a single prompt | Generating EN + PT-BR in one inference call keeps entity tags consistent and halves the compute cost vs. separate translation. |
| Human validation over additional LLM pass | With 7,231 consistent rows and a 10K target, the remaining edge cases are straightforward for human reviewers -- an additional LLM pass wasn't justified. |

### Models & Tools

| Component | Tool/Model | Purpose |
|---|---|---|
| Feature extraction | Gemma 3 12B (4-bit, Unsloth) | Structured multimodal image analysis |
| Entity translation | Llama 3.3 70B | Portuguese entity name extraction |
| Object detection | GroundingDINO Base | Bounding box generation from text prompts |
| Semantic parsing | LOME + DAISY | FrameNet frame/FE assignment |
| Framework | HuggingFace Transformers + bitsandbytes | Model loading, quantization, inference |
| Compute | CWRU HPC (SLURM), H100/L40s GPUs | GPU-accelerated batch processing |

### Code & Repository

The feature extraction scripts, HPC job configurations, and dataset download tools are available at: [GitHub - MultiModal_Framenet](https://github.com/neverbiasu/MultiModal_Framenet)

**Key scripts (Stage 1 -- Feature Extraction, in GitHub repo):**
- `Gemma_parallel_GPU2.py` -- Main multi-GPU feature extraction pipeline (auto-detects GPUs, round-robin assignment)
- `Gemma_parallel_GPU.py` / `Gemma_parallel_GPU1.py` -- Earlier 2-GPU parallel variants
- `Gemma_single_GPU.py` -- Single-GPU baseline version
- `gemma_analysis.py` -- Original sequential processing script
- `Download_dataset_extract.sh` -- Dataset download and extraction
- `config.sh` -- HPC environment setup (CUDA 12.1, cuDNN, Python modules)
- `request_node.sh` -- SLURM batch job submission (H100 GPU allocation)
- `interactive_node.sh` -- Interactive GPU node request for development/testing

**Stage 2 -- GroundingDINO & Coordinate Processing:**
- `grounding_dino_pipeline.py` -- Full GroundingDINO detection pipeline (standalone script, derived from Colab notebook `MultiModal_v1.ipynb`):
  - `process_dataset()` -- Batch processing of CSV dataset through GroundingDINO, producing bounding box coordinates and labeled images
  - `process_image_with_grounding_dino()` -- Per-image detection with coordinate scaling from processed to original image dimensions
  - `draw_boxes_on_image()` -- Bounding box visualization with labeled overlays
  - `validate_coordinates()` -- QA tool checking coordinate bounds against actual image dimensions

**Post-Processing Scripts:**
- `portuguese_entity_translation.py` -- Extracts Portuguese entity translations from existing Portuguese descriptions using Llama 3.3 70B, adds `Entity/Object List (Portuguese)` column, optionally fixes English entity consistency with description wording
