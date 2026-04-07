#!/usr/bin/env python
"""
grounding_dino_pipeline.py

Processes the feature extraction CSV through GroundingDINO to detect objects
in images using entity/object lists as text prompts. Produces bounding box
coordinates (both processed and original image space) and labeled images.

Usage:
    python grounding_dino_pipeline.py --input_csv structured_image_analysis_results.csv \
                                      --output_csv grounding_dino_results.csv \
                                      --output_images_dir detected_images
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
import io
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_entities(entity_list_str):
    """
    Parse the entity list string from the CSV into a list of entity strings.

    Handles comma, semicolon, and newline separators.
    """
    if pd.isna(entity_list_str) or not entity_list_str:
        return []

    entities = []
    if ',' in entity_list_str:
        entities = [e.strip() for e in entity_list_str.split(',') if e.strip()]
    elif ';' in entity_list_str:
        entities = [e.strip() for e in entity_list_str.split(';') if e.strip()]
    elif '\n' in entity_list_str:
        entities = [e.strip() for e in entity_list_str.split('\n') if e.strip()]
    else:
        entities = [entity_list_str.strip()]

    return entities


def process_image_with_grounding_dino(image, entities, processor, model, device,
                                      output_image_path=None,
                                      box_threshold=0.3, text_threshold=0.25):
    """
    Process an image with Grounding DINO to detect objects.

    Returns:
        tuple: (boxes_with_labels_processed, boxes_with_labels_original, output_path or None)
    """
    original_width, original_height = image.size

    inputs = processor(images=image, text=entities, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    processed_height, processed_width = inputs['pixel_values'].shape[-2:]

    target_sizes = torch.tensor([[processed_height, processed_width]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target_sizes,
        box_threshold=box_threshold, text_threshold=text_threshold
    )[0]

    x_scale = original_width / processed_width
    y_scale = original_height / processed_height

    boxes_with_labels_processed = []
    boxes_with_labels_original = []

    for box, text_label, score in zip(results["boxes"], results["text_labels"], results["scores"]):
        box_coords_processed = box.tolist()
        boxes_with_labels_processed.append({
            "label": text_label,
            "score": float(score),
            "box": box_coords_processed
        })

        x1, y1, x2, y2 = box_coords_processed
        box_coords_original = [
            x1 * x_scale, y1 * y_scale,
            x2 * x_scale, y2 * y_scale
        ]
        boxes_with_labels_original.append({
            "label": text_label,
            "score": float(score),
            "box": box_coords_original
        })

    labeled_image = draw_boxes_on_image(image, boxes_with_labels_original)

    if output_image_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        try:
            labeled_image.save(output_image_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"Error saving image to {output_image_path}: {str(e)}")

        return boxes_with_labels_processed, boxes_with_labels_original, output_image_path

    return boxes_with_labels_processed, boxes_with_labels_original, None


def draw_boxes_on_image(image, boxes_with_labels):
    """Draw bounding boxes on an image with labels."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(boxes_with_labels), 1)))

    for i, box_info in enumerate(boxes_with_labels):
        box = box_info["box"]
        label = box_info["label"]
        score = box_info["score"]

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor=colors[i % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 - 10,
            f"{label} ({score:.2f})",
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=colors[i % len(colors)], alpha=0.8, pad=2)
        )

    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)

    labeled_img = Image.open(buf)
    return labeled_img


def validate_coordinates(csv_path):
    """Validate coordinate data in the CSV file."""
    print("Validating coordinate data...")
    df = pd.read_csv(csv_path)
    issues = []

    for idx, row in df.iterrows():
        image_path = row['Image Path']

        if not os.path.exists(image_path):
            issues.append(f"Row {idx}: Image not found - {image_path}")
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            issues.append(f"Row {idx}: Cannot open image - {image_path}: {str(e)}")
            continue

        if pd.notna(row.get('Original_Image_Coordinates')) and row.get('Original_Image_Coordinates'):
            try:
                orig_boxes = json.loads(row['Original_Image_Coordinates'])
                for box_info in orig_boxes:
                    x1, y1, x2, y2 = box_info["box"]
                    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                        issues.append(
                            f"Row {idx}: Original coordinates out of bounds for {box_info['label']}"
                        )
            except Exception as e:
                issues.append(f"Row {idx}: Error parsing original coordinates: {str(e)}")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("All coordinate data appears valid!")

    return issues


def process_dataset(csv_path, output_csv_path, output_images_dir,
                    entity_column='Objects List',
                    box_threshold=0.3, text_threshold=0.25):
    """
    Process a dataset with Grounding DINO to detect objects in images.

    Args:
        csv_path: Path to the input CSV file (output from Gemma feature extraction)
        output_csv_path: Path to save the output CSV file
        output_images_dir: Directory to save the images with bounding boxes
        entity_column: Name of the column containing entity/object lists
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text matching
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Loading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print(f"Model loaded on {device}")

    os.makedirs(output_images_dir, exist_ok=True)

    df['Detected_Boxes_Image_Path'] = None
    df['Detected_Boxes_Coordinates'] = None
    df['Original_Image_Coordinates'] = None

    print("Processing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['Image Path']

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}, skipping...")
            continue

        entities = parse_entities(row.get(entity_column))
        if not entities:
            print(f"Warning: No valid entities found for row {idx}, skipping...")
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            image_filename = Path(image_path).name
            output_image_filename = f"detected_{Path(image_filename).stem}.jpg"
            output_image_path = os.path.join(output_images_dir, output_image_filename)

            boxes_coords, original_coords, _ = process_image_with_grounding_dino(
                image, entities, processor, model, device, output_image_path,
                box_threshold=box_threshold, text_threshold=text_threshold
            )

            df.at[idx, 'Detected_Boxes_Image_Path'] = output_image_path
            df.at[idx, 'Detected_Boxes_Coordinates'] = json.dumps(boxes_coords)
            df.at[idx, 'Original_Image_Coordinates'] = json.dumps(original_coords)

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

    print(f"Saving results to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    print(f"Processing complete! Labeled images saved to {output_images_dir}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process dataset with GroundingDINO for object detection'
    )
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV from feature extraction')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save output CSV with detection results')
    parser.add_argument('--output_images_dir', type=str, default='detected_images',
                        help='Directory to save labeled images')
    parser.add_argument('--entity_column', type=str, default='Objects List',
                        help='CSV column containing entity/object lists')
    parser.add_argument('--box_threshold', type=float, default=0.3,
                        help='Confidence threshold for bounding boxes')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Confidence threshold for text matching')
    parser.add_argument('--validate', action='store_true',
                        help='Run coordinate validation after processing')

    args = parser.parse_args()

    process_dataset(
        args.input_csv,
        args.output_csv,
        args.output_images_dir,
        entity_column=args.entity_column,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    if args.validate:
        print("\n" + "=" * 50)
        validate_coordinates(args.output_csv)
