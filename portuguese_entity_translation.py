#!/usr/bin/env python
"""
portuguese_entity_translation.py

Extracts Portuguese entity translations from existing Portuguese scene descriptions,
using Llama 3.3 70B (or a compatible model). Adds a new column with Portuguese
entity names matched to the English entity/object list.

Usage:
    python portuguese_entity_translation.py --input_csv grounding_dino_results.csv \
                                            --output_csv results_with_pt_entities.csv \
                                            --model meta-llama/Llama-3.3-70B-Instruct

    # Or with a smaller model for testing:
    python portuguese_entity_translation.py --input_csv grounding_dino_results.csv \
                                            --output_csv results_with_pt_entities.csv \
                                            --model meta-llama/Llama-3.1-8B-Instruct
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"

import pandas as pd
import torch
import argparse
import json
import re
from transformers import pipeline
from tqdm import tqdm


def build_translation_prompt(english_entities, portuguese_description):
    """
    Build a prompt that asks the LLM to extract Portuguese entity names
    from the Portuguese description that correspond to the English entities.
    """
    return f"""You are a bilingual entity extraction assistant. Given a list of English entity names and a Portuguese scene description, extract the corresponding Portuguese translation for each English entity from the description.

English entities: {english_entities}

Portuguese scene description: {portuguese_description}

For each English entity, find its Portuguese equivalent as it appears in the Portuguese description. Return ONLY a JSON object mapping English entities to their Portuguese translations. If an entity has no clear match in the Portuguese text, use a direct translation.

Output format (JSON only, no other text):
{{"english_entity_1": "portuguese_translation_1", "english_entity_2": "portuguese_translation_2"}}"""


def extract_portuguese_entities(english_entities_str, portuguese_description, pipe):
    """
    Use the LLM to extract Portuguese entity translations.

    Args:
        english_entities_str: Comma-separated English entities
        portuguese_description: Portuguese scene description
        pipe: Transformers text-generation pipeline

    Returns:
        str: Comma-separated Portuguese entity translations
    """
    if pd.isna(english_entities_str) or not english_entities_str.strip():
        return ""
    if pd.isna(portuguese_description) or not portuguese_description.strip():
        return ""

    prompt = build_translation_prompt(english_entities_str, portuguese_description)

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = pipe(
            messages,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1
        )

        generated_text = response[0]["generated_text"]
        if isinstance(generated_text, list):
            assistant_response = generated_text[-1]["content"]
        else:
            assistant_response = generated_text

        json_match = re.search(r'\{[^{}]+\}', assistant_response)
        if json_match:
            translation_map = json.loads(json_match.group())
            english_entities = [e.strip() for e in english_entities_str.split(',') if e.strip()]
            portuguese_entities = []
            for entity in english_entities:
                pt_name = translation_map.get(entity, entity)
                portuguese_entities.append(pt_name)
            return ', '.join(portuguese_entities)

        return ""

    except Exception as e:
        print(f"Error during translation: {e}")
        return ""


def fix_entity_consistency(entities_str, description):
    """
    Fix entity wording to match the exact phrasing used in the scene description.

    For example, if the entity list says "other player" but the description says
    "another player", update the entity to match the description.
    """
    if pd.isna(entities_str) or pd.isna(description):
        return entities_str

    entities = [e.strip() for e in entities_str.split(',') if e.strip()]
    description_lower = description.lower()
    fixed_entities = []

    for entity in entities:
        if entity.lower() in description_lower:
            fixed_entities.append(entity)
        else:
            words = entity.lower().split()
            best_match = entity
            best_overlap = 0

            sentences = re.split(r'[.!?]', description)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                overlap = sum(1 for w in words if w in sentence_lower)
                if overlap > best_overlap:
                    best_overlap = overlap
                    for word in words:
                        pattern = r'\b\w*' + re.escape(word) + r'\w*\b'
                        match = re.search(pattern, sentence_lower)
                        if match:
                            start = max(0, match.start() - 30)
                            end = min(len(sentence), match.end() + 30)
                            context = sentence[start:end].strip()
                            if len(words) == 1:
                                best_match = match.group()

            fixed_entities.append(best_match)

    return ', '.join(fixed_entities)


def process_dataset(input_csv, output_csv, model_name,
                    entity_column='Objects List',
                    pt_description_column='Scene Description (Portuguese)',
                    en_description_column='Scene Description (English)',
                    fix_consistency=True,
                    batch_size=1):
    """
    Process the dataset to add Portuguese entity translations.

    Args:
        input_csv: Path to input CSV
        output_csv: Path to output CSV
        model_name: HuggingFace model ID for the LLM
        entity_column: Column with English entity/object list
        pt_description_column: Column with Portuguese scene description
        en_description_column: Column with English scene description
        fix_consistency: Whether to fix entity wording consistency
        batch_size: Processing batch size
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    print(f"Loading model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded")

    entity_col_idx = df.columns.get_loc(entity_column)
    df.insert(entity_col_idx + 1, 'Entity/Object List (Portuguese)', '')

    if fix_consistency:
        print("Fixing entity consistency with descriptions...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fixing consistency"):
            if pd.notna(row.get(entity_column)) and pd.notna(row.get(en_description_column)):
                df.at[idx, entity_column] = fix_entity_consistency(
                    row[entity_column], row[en_description_column]
                )

    print("Extracting Portuguese entity translations...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Translating entities"):
        pt_entities = extract_portuguese_entities(
            row.get(entity_column, ''),
            row.get(pt_description_column, ''),
            pipe
        )
        df.at[idx, 'Entity/Object List (Portuguese)'] = pt_entities

        if (idx + 1) % 100 == 0:
            df.to_csv(output_csv, index=False)
            print(f"  Checkpoint saved at row {idx + 1}")

    print(f"Saving final results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Done! Added Portuguese entity translations for {len(df)} rows")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Portuguese entity translations using LLM'
    )
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save output CSV')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                        help='HuggingFace model ID')
    parser.add_argument('--entity_column', type=str, default='Objects List',
                        help='Column containing English entities')
    parser.add_argument('--pt_description_column', type=str,
                        default='Scene Description (Portuguese)',
                        help='Column containing Portuguese description')
    parser.add_argument('--en_description_column', type=str,
                        default='Scene Description (English)',
                        help='Column containing English description')
    parser.add_argument('--no_fix_consistency', action='store_true',
                        help='Skip entity consistency fixing')

    args = parser.parse_args()

    process_dataset(
        args.input_csv,
        args.output_csv,
        args.model,
        entity_column=args.entity_column,
        pt_description_column=args.pt_description_column,
        en_description_column=args.en_description_column,
        fix_consistency=not args.no_fix_consistency
    )
