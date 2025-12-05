"""Simple demo script to test Japanese pattern recognizers without GiNZA."""

from recognizers.japanese_patterns import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
)

# Sample Japanese resume text
sample_text = """
履歴書

氏名: 山田太郎
ふりがな: やまだたろう

生年月日: 1990年1月1日 (34歳)

現住所:
〒150-0001
東京都渋谷区神宮前1-2-3 サンプルマンション101号室

連絡先:
TEL: 03-1234-5678
携帯: 090-9876-5432
Email: yamada.taro@example.com
"""


def test_pattern_recognizers():
    """Test pattern-based recognizers."""
    print("=" * 60)
    print("Testing Japanese Pattern Recognizers")
    print("=" * 60)
    
    # Test phone recognizer
    print("\n1. Phone Number Recognizer")
    print("-" * 40)
    phone_recognizer = JapanesePhoneRecognizer()
    phone_results = phone_recognizer.analyze(sample_text, entities=["PHONE_NUMBER_JP"])
    
    if phone_results:
        for i, result in enumerate(phone_results, 1):
            detected_text = sample_text[result.start:result.end]
            print(f"  {i}. Found: '{detected_text}' (score: {result.score:.2f})")
    else:
        print("  No phone numbers detected")
    
    # Test zip code recognizer
    print("\n2. Zip Code Recognizer")
    print("-" * 40)
    zip_recognizer = JapaneseZipCodeRecognizer()
    zip_results = zip_recognizer.analyze(sample_text, entities=["JP_ZIP_CODE"])
    
    if zip_results:
        for i, result in enumerate(zip_results, 1):
            detected_text = sample_text[result.start:result.end]
            print(f"  {i}. Found: '{detected_text}' (score: {result.score:.2f})")
    else:
        print("  No zip codes detected")
    
    # Test birth date recognizer
    print("\n3. Birth Date Recognizer")
    print("-" * 40)
    date_recognizer = JapaneseBirthDateRecognizer()
    date_results = date_recognizer.analyze(sample_text, entities=["DATE_OF_BIRTH_JP"])
    
    if date_results:
        for i, result in enumerate(date_results, 1):
            detected_text = sample_text[result.start:result.end]
            print(f"  {i}. Found: '{detected_text}' (score: {result.score:.2f})")
    else:
        print("  No birth dates detected")
    
    # Summary
    print("\n" + "=" * 60)
    total = len(phone_results) + len(zip_results) + len(date_results)
    print(f"Total entities detected: {total}")
    print("=" * 60)


if __name__ == "__main__":
    test_pattern_recognizers()
