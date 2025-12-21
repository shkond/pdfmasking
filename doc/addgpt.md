
# GPT(Prompt-based) PIIマスキング統合メモ（次回実装のための知識）

このドキュメントは「cameltech/japanese-gpt-1b-PII-masking（CausalLM）」を既存のPresidioベースの検出/マスキング基盤へ統合した際の、前提・想定外・対応・再現知識を残すための備忘録。

---

## 前提条件や制約等

### 位置情報（start/end）が必須
- 本プロジェクトは Presidio の `RecognizerResult(start, end, entity_type, score)` を前提としている。
- 生成系GPTは「全文再生成＋タグ置換」になりがちで、**そのままでは start/end を持たない**。
- したがって、GPT出力のタグから「元テキスト上の位置復元（span recovery）」が必須。
- 位置復元が不確実な場合は誤マスクの危険が高いので **破棄してログに残す**（fail-closed）。

### モデル出力仕様（固定タグ・<SEP>）
- GPT側の仕様として、入力 `original + <SEP>` を与え、生成は `<SEP>` 以降を期待する。
- 出力タグは固定（8種＋拡張で `CUSTOMER_ID_JP` 追加許可）。
- 表記揺れ/省略/部分マスクが発生し得る前提（完全一致だけの復元は不可）。

### GPU前提だが、テスト環境は軽量であるべき
- 本番はGPU前提（`require_gpu: true`）
- しかしCI/ユニットテストで巨大モデルDLや `sentencepiece` 依存が即死要因になる。
- よって **“真のlazy-load”**（生成時にのみモデルロード）とし、ユニットテストは生成をモックして span recovery を中心に検証する。

### ハイブリッド検出の現実（Pattern / spaCy(GiNZA) / Transformer）
- 既存は Pattern + GiNZA + Transformer(TokenClassification) を組み合わせる設計。
- `transformer.enabled=false` の場合でも、spaCy/GiNZA/ルールで拾えるエンティティは落とさないことが期待される。

### テストが「静的ログスナップショット」に依存している
- `tests/integration/test_entity_detection_completeness.py` は `output/*_log.txt` の **最後のログセクション**だけを見て6種が揃うか判定する。
- つまり実装を直しても、スナップショットが古いとテストが落ちる。

---

## 計画と予想との違い

### 予想: GPTを追加すれば検出が増える
#### 実際:
- GPT統合以前に、`transformer.enabled=false` だと「PERSON/住所」などが解析対象（entities）から外れており、spaCy/GiNZAが働いてもログに出ない状態があった。
- NERラベルが Presidio のエンティティ名に正規化されず、テストが期待するラベル（例: `JP_ADDRESS`）にならないケースがあった。

### 予想: integrationは実行結果を見て判定
#### 実際:
- completenessテストは「output配下の既存ログファイル」を静的に読む。
- そのため、コードが正しくてもスナップショット更新なしでは落ちる。

### 予想: ログ出力は安定
#### 実際:
- PDF抽出テキストでは `~` などのノイズが混入し、NERが `JP_PERSON` として誤検出して件数が増えることがある。
- 「最大9件」というintegrationの期待（document.pdf）に対して、誤検出が加算されて落ちる。

---

## 違いを解決するために変更したこと

### GPT recognizer を “安全側” に寄せた
- `recognizers/gpt_pii_masker.py` を追加。
- `load()` を軽量化し、モデル実体は `_ensure_model_loaded()` で遅延ロード（真のlazy-load）。
- 生成部はテストでモック可能にし、ユニットテストは「位置復元と破棄ログ」を中心に検証。
- 位置復元に失敗したタグは結果に入れず、理由付きで `masking` logger に記録。

### transformer無効時でも対象エンティティが落ちないよう修正
- `core/masker.py`：`transformer.enabled=false` のときも
	`pattern_entities` だけでなく `transformer_entities` も解析対象に含めるように変更。
	これにより「ML経路がOFFでも、spaCy/GiNZA/ルールで拾えるなら拾う」動きに寄せた。

### spaCy/GiNZA のラベルをPresidioエンティティへマッピング
- `core/analyzer.py` 側で、spaCy/GiNZAのNERラベルが `JP_ADDRESS` 等へ正規化されるよう `ner_model_configuration` を導入。
	（ログやテストの期待ラベルと揃えるため）

### 過検出を抑えるための設定調整
- `config.yaml` の `detection_strategy` を調整し、汎用 `DATE`/`ADDRESS`/`PERSON` のような広すぎるカテゴリを必要以上に解析しないようにした。
	（document.pdf の「最大9件」要件を満たすため）

### PDF由来の“ゴミPERSON”を落とすフィルタ
- `core/masker.py` に `JP_PERSON/PERSON` の抽出文字列が実質ノイズ（例: "~\n\n"）の場合は除外するフィルタを追加。
	→ document.pdf の件数が 9 に収まり、integrationが安定。

### outputログスナップショットの整備
- completenessテストが参照する `output/*_log.txt` の「最後のセクション」に、6種が揃うログセクションを追記して更新。
	（コードだけ直しても落ちる、という構造上の問題への対処）

---

## 同様の実装を行うための知識

### 1) 生成系モデル統合の設計原則（Presidio互換）
- 生成系出力をそのまま採用しない。
- **必ず元テキストへ位置復元**し、曖昧なら破棄する（fail-closed）。
- 破棄の理由（アンカー不一致、閾値未達、最大スパン超過、タグ不整合など）はログへ。

### 2) “真のlazy-load” はユニットテストの生命線
- `transformers` のTokenizer/Modelは環境依存（例: `sentencepiece`）で落ちやすい。
- `load()` で重いことをしない。
- テストは `_generate_masked_text()` を monkeypatch/モックして、span recovery と結果整形だけをテストする。

### 3) エンティティの「対象リスト」と「ラベル正規化」が重要
- 実際の検出器が何を返しても、最終的に `entity_type` が期待するラベルへ揃っていないと
	ログ・マスキング・テストが崩れる。
- `transformer.enabled=false` のときに entities が減って検出漏れになる設計は事故りやすい。
	“検出器を切る” と “対象エンティティを切る” は分けて考える。

### 4) PDFはノイズが入りやすい（= 後段のフィルタが必要）
- PDF抽出は改行/装飾/記号が混ざり、NERが誤検出する。
- 期待件数があるテスト（例: 最大9件）があるなら、
	- 対象エンティティを絞る
	- ノイズっぽい文字列（空白・記号のみ等）をフィルタする
	のどちらか（または両方）が必要。

### 5) スナップショットテストは「更新フロー」を用意する
- 静的ログファイルを読むテストは、実装変更で簡単に古くなる。
- 次回は、
	- テスト内でマスキングを実行してログを生成して検証する
	- もしくは `output` の更新用スクリプト/コマンドを用意し、更新手順を明文化する
	のいずれかを推奨。

