#!/usr/bin/env python3
"""
Resume PDF Generator for Machine Learning Training Dataset
Generates PDF files from resume data JSON

Usage:
    python generate_pdf.py

Requirements:
    pip install reportlab Pillow
"""

import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from datetime import datetime
import os

class ResumeGenerator:
    def __init__(self, output_dir="resumes_pdf"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a4d7f'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=3,
            fontName='Helvetica-Bold',
            spaceBefore=6
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=9,
            leading=11
        ))

    def generate_pdf(self, resume_data):
        """Generate PDF from resume data"""
        resume_id = resume_data['id']
        language = resume_data['language']

        # Determine page size and filename
        page_size = letter if language == "English" else A4
        filename = f"{self.output_dir}/{resume_id}_{resume_data['job_title'].replace(' ', '_')}.pdf"

        # Create document
        doc = SimpleDocTemplate(filename, pagesize=page_size)
        elements = []

        # Header with personal info
        personal_info = resume_data['personal_info']

        # Name
        elements.append(Paragraph(
            personal_info['name'],
            self.styles['CustomHeading1']
        ))

        # Contact info
        contact_lines = [
            f"Email: {personal_info['email']}",
            f"Phone: {personal_info['phone']}",
            f"Address: {personal_info['address']}",
            f"Postal Code: {personal_info['postal_code']}",
            f"Date of Birth: {personal_info['date_of_birth']}"
        ]

        for line in contact_lines:
            elements.append(Paragraph(line, self.styles['CustomBody']))

        elements.append(Spacer(1, 0.2 * inch))

        # Professional Summary
        job_title = resume_data['job_title']
        if language == "English":
            summary = f"Experienced {job_title} with expertise in {', '.join(resume_data['technologies'][:3])}."
        else:
            summary = f"{job_title}として、{', '.join(resume_data['technologies'][:3])}の知識を有する。"

        elements.append(Paragraph(
            "PROFESSIONAL SUMMARY" if language == "English" else "職務経歴書サマリー",
            self.styles['CustomHeading2']
        ))
        elements.append(Paragraph(summary, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.1 * inch))

        # Education
        education = resume_data['education']
        elements.append(Paragraph(
            "EDUCATION" if language == "English" else "学歴",
            self.styles['CustomHeading2']
        ))

        edu_text = f"{education['university']} - {education['degree']} ({education['graduation_year']})"
        elements.append(Paragraph(edu_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.1 * inch))

        # Experience
        elements.append(Paragraph(
            "WORK EXPERIENCE" if language == "English" else "職務経歴",
            self.styles['CustomHeading2']
        ))

        for exp in resume_data['work_experience']:
            exp_title = f"{exp['company']} - {exp['position']} ({exp['start_year']}-{exp['end_year'] or '現在'})"
            elements.append(Paragraph(exp_title, self.styles['CustomBody']))
            elements.append(Paragraph(exp['description'], self.styles['CustomBody']))
            elements.append(Spacer(1, 0.05 * inch))

        elements.append(Spacer(1, 0.1 * inch))

        # Skills/Technologies
        elements.append(Paragraph(
            "TECHNICAL SKILLS" if language == "English" else "技術スキル",
            self.styles['CustomHeading2']
        ))

        tech_text = ", ".join(resume_data['technologies'])
        elements.append(Paragraph(tech_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.1 * inch))

        # Certifications
        elements.append(Paragraph(
            "CERTIFICATIONS" if language == "English" else "資格",
            self.styles['CustomHeading2']
        ))

        for cert in resume_data['certifications']:
            elements.append(Paragraph(f"• {cert}", self.styles['CustomBody']))

        # Build PDF
        doc.build(elements)

        return filename

    def generate_all(self, resume_data_file="resume_data.json"):
        """Generate PDFs for all resumes"""
        with open(resume_data_file, 'r', encoding='utf-8') as f:
            resumes = json.load(f)

        generated_files = []
        for resume in resumes:
            filename = self.generate_pdf(resume)
            generated_files.append(filename)
            print(f"✓ Generated: {filename}")

        print(f"\n✓ Successfully generated {len(generated_files)} PDF files in '{self.output_dir}' directory")
        return generated_files


if __name__ == "__main__":
    print("Resume PDF Generator")
    print("=" * 50)

    try:
        generator = ResumeGenerator()
        generator.generate_all()
        print("\n✓ All PDFs generated successfully!")
    except FileNotFoundError:
        print("✗ Error: resume_data.json not found")
        print("Please ensure resume_data.json is in the current directory")
    except Exception as e:
        print(f"✗ Error: {e}")
