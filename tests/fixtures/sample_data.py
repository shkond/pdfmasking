"""Sample Japanese resume text for testing."""

SAMPLE_RESUME_TEXT = """
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

学歴:
2008年4月 東京大学 工学部 入学
2012年3月 東京大学 工学部 卒業

職歴:
2012年4月 株式会社サンプル 入社
2015年4月 同社 開発部 配属
2020年3月 同社 退社

資格:
2013年6月 基本情報技術者試験 合格
2015年12月 応用情報技術者試験 合格

志望動機:
貴社の革新的な技術開発に魅力を感じ、応募いたしました。

以上
"""


def get_sample_resume() -> str:
    """Get sample Japanese resume text."""
    return SAMPLE_RESUME_TEXT


def get_expected_entities() -> list:
    """
    Get expected PII entities in the sample resume.
    
    Returns:
        List of dicts with entity information
    """
    return [
        {"type": "JP_PERSON", "text": "山田太郎"},
        {"type": "JP_ADDRESS", "text": "東京都渋谷区神宮前1-2-3"},
        {"type": "PHONE_NUMBER_JP", "text": "03-1234-5678"},
        {"type": "PHONE_NUMBER_JP", "text": "090-9876-5432"},
        {"type": "JP_ZIP_CODE", "text": "150-0001"},
        {"type": "DATE_OF_BIRTH_JP", "text": "1990年1月1日"},
        {"type": "EMAIL_ADDRESS", "text": "yamada.taro@example.com"},
    ]
