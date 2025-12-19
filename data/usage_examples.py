#!/usr/bin/env python3
"""
Usage Examples: Resume Dataset for Machine Learning

This script demonstrates how to use the resume dataset for various NLP/ML tasks.
"""

import json
from collections import Counter

def load_data():
    """Load all dataset files"""
    with open('resume_data.json', 'r', encoding='utf-8') as f:
        resumes = json.load(f)

    with open('annotations.json', 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    with open('normalized_data.json', 'r', encoding='utf-8') as f:
        normalized = json.load(f)

    return resumes, annotations, normalized

def example_1_extract_emails():
    """Example 1: Extract all emails by language"""
    resumes, _, _ = load_data()

    print("Example 1: Extract Emails by Language")
    print("=" * 50)

    english_emails = [r['personal_info']['email'] for r in resumes if r['language'] == 'English']
    japanese_emails = [r['personal_info']['email'] for r in resumes if r['language'] == 'Japanese']

    print(f"\nEnglish emails ({len(english_emails)}):")
    for email in english_emails:
        print(f"  - {email}")

    print(f"\nJapanese emails ({len(japanese_emails)}):")
    for email in japanese_emails:
        print(f"  - {email}")

def example_2_normalize_data():
    """Example 2: Use normalized data for validation"""
    _, _, normalized = load_data()

    print("\n\nExample 2: Normalized Data Processing")
    print("=" * 50)

    for resume in normalized[:2]:  # Show first 2 resumes
        print(f"\nResume ID: {resume['resume_id']} ({resume['language']})")
        print("\nNormalized Personal Info:")

        name_info = resume['normalized_personal_info']['name']
        print(f"  Name: {name_info['original']}")
        if name_info['format'] == 'WESTERN':
            print(f"    First: {name_info['first_name']}, Last: {name_info['last_name']}")

        email_info = resume['normalized_personal_info']['email']
        print(f"  Email: {email_info['original']}")
        print(f"    Domain: {email_info.get('domain', 'N/A')}")

        phone_info = resume['normalized_personal_info']['phone']
        print(f"  Phone: {phone_info['original']}")
        print(f"    Country: {phone_info.get('country_code', 'N/A')}")

        dob_info = resume['normalized_personal_info']['date_of_birth']
        print(f"  DOB: {dob_info['original']}")
        print(f"    Age: {dob_info['age']}")

def example_3_job_title_distribution():
    """Example 3: Analyze job title distribution"""
    resumes, _, _ = load_data()

    print("\n\nExample 3: Job Title Distribution")
    print("=" * 50)

    job_titles = Counter([r['job_title'] for r in resumes])

    print("\nJob titles in dataset:")
    for title, count in job_titles.most_common():
        print(f"  {title}: {count}")

def example_4_technology_extraction():
    """Example 4: Extract and count technologies"""
    resumes, _, _ = load_data()

    print("\n\nExample 4: Technology Stack Analysis")
    print("=" * 50)

    all_techs = []
    for resume in resumes:
        all_techs.extend(resume['technologies'])

    tech_count = Counter(all_techs)

    print("\nMost common technologies:")
    for tech, count in tech_count.most_common(10):
        print(f"  {tech}: {count}")

def example_5_annotation_extraction():
    """Example 5: Work with annotations"""
    _, annotations, _ = load_data()

    print("\n\nExample 5: Annotation Structure")
    print("=" * 50)

    entity_types = set()
    for annotation in annotations:
        for key, value in annotation['annotations'].items():
            entity_types.add(value['type'])

    print("\nEntity types in dataset:")
    for entity_type in sorted(entity_types):
        print(f"  - {entity_type}")

    print("\nSample annotations (First resume):")
    sample = annotations[0]
    for key, annotation in sample['annotations'].items():
        print(f"\n  {key}:")
        print(f"    Type: {annotation['type']}")
        print(f"    Value: {annotation['value']}")

def example_6_language_comparison():
    """Example 6: Compare English vs Japanese data"""
    resumes, _, normalized = load_data()

    print("\n\nExample 6: Language Comparison")
    print("=" * 50)

    en_resumes = [r for r in resumes if r['language'] == 'English']
    jp_resumes = [r for r in resumes if r['language'] == 'Japanese']

    print(f"\nEnglish resumes: {len(en_resumes)}")
    for r in en_resumes:
        print(f"  ID: {r['id']}, Title: {r['job_title']}")

    print(f"\nJapanese resumes: {len(jp_resumes)}")
    for r in jp_resumes:
        print(f"  ID: {r['id']}, Title: {r['job_title']}")

    # Check normalization differences
    print("\nNormalization differences:")
    en_norm = [n for n in normalized if n['language'] == 'English'][0]
    jp_norm = [n for n in normalized if n['language'] == 'Japanese'][0]

    print(f"\nEnglish postal code format:")
    print(f"  {en_norm['normalized_personal_info']['postal_code']}")

    print(f"\nJapanese postal code format:")
    print(f"  {jp_norm['normalized_personal_info']['postal_code']}")

if __name__ == "__main__":
    try:
        example_1_extract_emails()
        example_2_normalize_data()
        example_3_job_title_distribution()
        example_4_technology_extraction()
        example_5_annotation_extraction()
        example_6_language_comparison()

        print("\n\n✓ All examples completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all JSON files are in the current directory")
