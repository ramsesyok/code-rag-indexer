# ソースコードRAG登録ツール 要件定義書（Go言語版）

**文書バージョン**: 2.0  
**作成日**: 2025-11-23  
**更新日**: 2025-11-23  
**プロジェクト名**: Code RAG Indexer (Go Implementation)

---

## 1. プロジェクトの概要

### 1.1 背景と目的

本プロジェクトは、ローカルLLMとローカルRAGを活用したソースコード理解支援システムの構築を目的とする。チーム開発において、大規模なコードベース（10万行～100万行超）の理解、リファクタリング支援、バグ調査を効率化するため、AST（抽象構文木）解析に基づいた高精度なコード検索基盤を提供する。

### 1.2 システムの位置づけ

本ツールは、ソースコードをQdrantベクトルデータベースに登録するインデクサーとして機能する。登録されたデータは、別途構築されるLLMベースの問い合わせシステムから参照される。

### 1.3 適用範囲

- **対象ユーザー**: ソフトウェア開発チーム
- **対象コード**: C、C++、Java、Rust、Go、Python で記述されたソースコード
- **利用環境**: Windows、Linux上で動作するCLIツール
- **実装言語**: Go 1.21以上
- **補助サービス**: Python埋め込みサーバー（gRPC通信）

### 1.4 アーキテクチャ概要

```
┌─────────────────────────────────────┐
│   Go メインプログラム               │
│   (code-rag-indexer)                │
│                                     │
│  ┌──────────┐  ┌──────────────┐   │
│  │ AST      │  │ File Scanner │   │
│  │ Parser   │  │              │   │
│  │(Tree-    │  │              │   │
│  │ sitter)  │  │              │   │
│  └──────────┘  └──────────────┘   │
│         │              │            │
│         v              v            │
│  ┌──────────────────────────────┐  │
│  │   gRPC Client                │  │
│  │   (Embedding Request)        │  │
│  └──────────────────────────────┘  │
└─────────────┼───────────────────────┘
              │ gRPC
              v
┌─────────────────────────────────────┐
│   Python 埋め込みサーバー           │
│   (embedding-server)                │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   gRPC Server                │  │
│  │   (Embedding Service)        │  │
│  └──────────────────────────────┘  │
│              │                      │
│              v                      │
│  ┌──────────────────────────────┐  │
│  │  Jina Embeddings v2 Code     │  │
│  │  (PyTorch Model)             │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────┐
│   Qdrant Vector Database            │
│   (Go Client経由で接続)             │
└─────────────────────────────────────┘
```

---

## 2. 機能要件

### 2.1 入力機能

#### 2.1.1 ソースコード入力

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IN-001 | ローカルディレクトリを入力として受け付けること | 必須 | 初期実装対象 |
| IN-002 | Gitリポジトリをクローンして入力として受け付けること | 将来 | Phase 2で実装 |
| IN-003 | 設定ファイル（YAML形式）から入力パスを読み込めること | 必須 | |

#### 2.1.2 ファイル除外機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IN-101 | .ragignoreファイルで除外パターンを指定できること | 必須 | .gitignore互換の文法 |
| IN-102 | バイナリファイルを自動検出して除外すること | 必須 | |
| IN-103 | 対象外の拡張子を設定で指定できること | 必須 | テストコード、自動生成コードの除外に使用 |

### 2.2 AST構造分析機能

#### 2.2.1 コード解析

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| AST-001 | 対象言語（C/C++/Java/Rust/Go/Python）のソースコードをAST解析すること | 必須 | Tree-sitter使用 |
| AST-002 | 関数・メソッド単位でコードを分割すること | 必須 | RAGのチャンク単位 |
| AST-003 | 構文エラーのあるファイルは警告を出力してスキップすること | 必須 | パース不可能なファイル |
| AST-004 | 意味的に不完全なコードは警告を出力して登録すること | 必須 | 開発途中のコードも対象 |

#### 2.2.2 抽出情報

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| AST-101 | 関数名、引数、戻り値の型を抽出すること | 必須 | |
| AST-102 | クラス階層、継承関係を抽出すること | 必須 | |
| AST-103 | 依存関係（import/include）を抽出すること | 必須 | |
| AST-104 | コメント、ドキュメント文字列を抽出すること | 必須 | |
| AST-105 | スコープ情報（グローバル/クラスメンバー/ローカル）を抽出すること | 必須 | |
| AST-106 | 修飾子（public/private/protected/static/const/async等）を抽出すること | 必須 | |
| AST-107 | 位置情報（開始行・終了行・開始列・終了列）を抽出すること | 必須 | エラー箇所特定に使用 |
| AST-108 | 呼び出し関係（この関数が呼び出す他の関数）を抽出すること | 必須 | |
| AST-109 | 被呼び出し関係（この関数を呼び出す関数）を抽出すること | 推奨 | 逆引き用、要追加解析 |
| AST-110 | 使用している型の一覧を抽出すること | 必須 | |
| AST-111 | サイクロマティック複雑度を計算すること | 推奨 | バグ潜在箇所の特定に有用 |
| AST-112 | 実効行数、コメント行数を計算すること | 推奨 | |
| AST-113 | 言語固有の属性（デコレータ/アノテーション/属性等）を抽出すること | 推奨 | |

### 2.3 ベクトル化・登録機能

#### 2.3.1 ベクトル化（gRPC経由）

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| VEC-001 | Python埋め込みサーバーとgRPCで通信すること | 必須 | HTTP/RESTではなくgRPC |
| VEC-002 | jinaai/jina-embeddings-v2-base-codeモデル（768次元）を使用してコードをベクトル化すること | 必須 | サーバー側で実行 |
| VEC-003 | 最大8192トークンまでのコンテキストを処理できること | 必須 | 長い関数・クラスに対応 |
| VEC-004 | コード本体とコメントを結合してベクトル化すること | 必須 | |
| VEC-005 | バッチリクエストに対応すること | 必須 | 複数コードを一度に送信 |
| VEC-006 | gRPC接続エラー時にリトライすること | 必須 | 3回までリトライ |
| VEC-007 | 埋め込みサーバーのヘルスチェックを実行できること | 必須 | 起動前の確認 |
| VEC-008 | 周辺コンテキストを含めてベクトル化できること | 将来 | Phase 2で実装 |
| VEC-009 | MCPサーバー（検索側）と同一のモデル・バージョンを使用すること | 必須 | ベクトル空間の一致が必要 |

#### 2.3.2 Qdrantへの登録

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| QDR-001 | Qdrantに関数・メソッド単位でベクトルを登録すること | 必須 | |
| QDR-002 | ファイルパスをメタデータとして保存すること | 必須 | |
| QDR-003 | 行番号（開始・終了）をメタデータとして保存すること | 必須 | |
| QDR-004 | プログラミング言語名をメタデータとして保存すること | 必須 | |
| QDR-005 | 構文要素の種類（関数/メソッド/クラス等）をメタデータとして保存すること | 必須 | |
| QDR-006 | リポジトリ情報（名前/バージョン/コミットハッシュ）をメタデータとして保存すること | 必須 | |
| QDR-007 | AST解析で抽出した全メタ情報をメタデータとして保存すること | 必須 | 検索・フィルタリングに使用 |

### 2.4 コレクション管理機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| COL-001 | コレクション名をカスタム指定できること | 必須 | 設定ファイルで指定 |
| COL-002 | コレクション名をディレクトリ名ベースで自動生成できること | 必須 | デフォルト動作 |
| COL-003 | リポジトリ名ベースでコレクション名を自動生成できること | 将来 | Git対応時に実装 |
| COL-004 | 既存コレクションが存在する場合、削除して再作成すること | 必須 | 上書き動作 |
| COL-005 | リポジトリごとに独立したコレクションを作成すること | 必須 | |

### 2.5 設定管理機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| CFG-001 | YAML形式の設定ファイルを読み込めること | 必須 | |
| CFG-002 | 入力ディレクトリパスを設定できること | 必須 | |
| CFG-003 | Qdrant接続情報（URL、APIキー）を設定できること | 必須 | |
| CFG-004 | 埋め込みサーバー接続情報（gRPCアドレス）を設定できること | 必須 | 新規追加 |
| CFG-005 | コレクション名を設定できること | 必須 | |
| CFG-006 | 除外パターンファイル（.ragignore）のパスを設定できること | 必須 | |
| CFG-007 | 埋め込みモデルの設定（モデル名、最大トークン長）を設定できること | 必須 | サーバー側設定の参照用 |
| CFG-008 | 並列処理数を設定できること | 推奨 | |
| CFG-009 | ログレベル（DEBUG/INFO/WARN/ERROR）を設定できること | 推奨 | |

### 2.6 埋め込みサーバー機能

#### 2.6.1 サーバー側機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| EMB-001 | gRPCサーバーとして起動すること | 必須 | Python実装 |
| EMB-002 | Jina Embeddings v2 Codeモデルを読み込むこと | 必須 | 起動時に1回 |
| EMB-003 | 単一コードのベクトル化リクエストに対応すること | 必須 | EmbedSingle RPC |
| EMB-004 | バッチコードのベクトル化リクエストに対応すること | 必須 | EmbedBatch RPC |
| EMB-005 | ヘルスチェックリクエストに対応すること | 必須 | Health RPC |
| EMB-006 | GPU/CPU自動切り替えに対応すること | 必須 | CUDA利用可能時はGPU |
| EMB-007 | エラー時は適切なgRPCステータスコードを返すこと | 必須 | INVALID_ARGUMENT等 |
| EMB-008 | リクエストログを出力すること | 推奨 | デバッグ用 |

#### 2.6.2 クライアント側機能（Go）

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| EMB-101 | gRPCクライアントを初期化すること | 必須 | 起動時 |
| EMB-102 | 埋め込みサーバーへの接続を確立すること | 必須 | TLS未使用（ローカル通信） |
| EMB-103 | ヘルスチェックでサーバーの稼働を確認すること | 必須 | メイン処理前 |
| EMB-104 | バッチリクエストでベクトル化を依頼すること | 必須 | 効率化 |
| EMB-105 | 接続エラー時にリトライすること | 必須 | 最大3回 |
| EMB-106 | タイムアウトを設定すること | 必須 | デフォルト30秒 |
| EMB-107 | コンテキストキャンセルに対応すること | 必須 | Ctrl+C等 |

---

## 3. 非機能要件

### 3.1 性能要件

| 要件ID | 要件内容 | 目標値 | 備考 |
|--------|---------|--------|------|
| PRF-001 | ファイル単位で並列処理を行うこと | - | Goroutineで並列化 |
| PRF-002 | 16GB～32GBのメモリで動作すること | 32GB以下 | 一般的なPC環境 |
| PRF-003 | 100万行規模のコードベースを一晩（8時間以内）で処理できること | 8時間以内 | PoCレベル目標 |
| PRF-004 | gRPC通信のオーバーヘッドを最小化すること | - | バッチリクエスト活用 |

### 3.2 可用性・信頼性要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| REL-001 | 一部ファイルの解析失敗時も、処理を継続すること | 警告を出力してスキップ |
| REL-002 | 処理中断時、どのファイルまで処理したか記録すること | ログから追跡可能にする |
| REL-003 | 埋め込みサーバーダウン時はエラー終了すること | 自動復旧は行わない |
| REL-004 | Qdrant接続断時はリトライすること | 最大3回 |

### 3.3 保守性・拡張性要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| MNT-001 | 新しいプログラミング言語の追加が容易な設計とすること | パーサーの追加のみで対応 |
| MNT-002 | 埋め込みモデルの変更が容易な設計とすること | サーバー側の設定変更のみ |
| MNT-003 | シングルバイナリで配布できること | Go単体でコンパイル可能 |
| MNT-004 | MCPサーバーとモデル設定を共有できること | 設定ファイルの共通化 |
| MNT-005 | 埋め込みサーバーを独立して起動・停止できること | サービス化対応 |

### 3.4 移植性要件

| 要件ID | 要件内容 | 対象環境 |
|--------|---------|---------|
| PRT-001 | Windows環境で動作すること | Windows 10以降 |
| PRT-002 | Linux環境で動作すること | Ubuntu 20.04以降、RHEL 8以降 |
| PRT-003 | Go 1.21以上で動作すること | 型パラメータ等の機能を活用 |
| PRT-004 | Python 3.11以上で動作すること | 埋め込みサーバー |

### 3.5 セキュリティ要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| SEC-001 | Qdrant接続時、APIキー認証を使用すること | 設定ファイルから読み込み |
| SEC-002 | APIキーを環境変数から読み込めること | 設定ファイルへの平文保存を回避 |
| SEC-003 | gRPC通信はTLSを使用しないこと | ローカル通信のみ想定 |
| SEC-004 | 機密情報のフィルタリングは行わないこと | ユーザー責任で管理 |

### 3.6 運用要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| OPR-001 | CLIインターフェースを提供すること | GUI不要 |
| OPR-002 | ログをファイルとコンソールに出力すること | 両方に同時出力 |
| OPR-003 | ファイル単位の読み込み開始を通知すること | 進捗確認用 |
| OPR-004 | 警告・エラーを明確に区別してログ出力すること | |
| OPR-005 | 処理完了時、登録件数のサマリーを表示すること | |
| OPR-006 | 埋め込みサーバーを別プロセスで起動すること | systemdやsupervisor推奨 |
| OPR-007 | 埋め込みサーバーのログを独立して管理すること | サーバー側でログファイル出力 |

---

## 4. インターフェース要件

### 4.1 コマンドラインインターフェース（Go）

```bash
code-rag-indexer [OPTIONS]
```

#### オプション

| オプション | 説明 | 必須/任意 |
|----------|------|---------|
| `-c, --config <PATH>` | 設定ファイルのパス | 必須 |
| `-v, --verbose` | 詳細ログ出力 | 任意 |
| `-h, --help` | ヘルプ表示 | 任意 |
| `--version` | バージョン表示 | 任意 |

### 4.2 埋め込みサーバー起動コマンド（Python）

```bash
embedding-server [OPTIONS]
```

#### オプション

| オプション | 説明 | 必須/任意 | デフォルト |
|----------|------|---------|-----------|
| `--host <HOST>` | バインドするホスト | 任意 | localhost |
| `--port <PORT>` | バインドするポート | 任意 | 50051 |
| `--model <MODEL>` | モデル名 | 任意 | jinaai/jina-embeddings-v2-base-code |
| `--max-length <INT>` | 最大トークン長 | 任意 | 8192 |
| `--batch-size <INT>` | バッチサイズ | 任意 | 32 |
| `--log-level <LEVEL>` | ログレベル | 任意 | INFO |
| `--log-file <PATH>` | ログファイルパス | 任意 | embedding-server.log |

### 4.3 設定ファイル仕様

**ファイル形式**: YAML

```yaml
# 入力設定
input:
  source_dir: "/path/to/source/code"
  ignore_file: ".ragignore"

# Qdrant設定
qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"
  collection_name: "my-project"

# 埋め込みサーバー設定（新規追加）
embedding_server:
  address: "localhost:50051"  # gRPCアドレス
  timeout: 30s                # リクエストタイムアウト
  max_retries: 3              # 最大リトライ回数
  batch_size: 8               # バッチリクエストサイズ

# 埋め込みモデル設定（参照用）
embedding:
  model_name: "jinaai/jina-embeddings-v2-base-code"
  dimension: 768
  max_length: 8192

# 処理設定
processing:
  parallel_workers: 4
  languages:
    - python
    - rust
    - go
    - java
    - c
    - cpp

# ログ設定
logging:
  level: "INFO"
  file: "code-rag-indexer.log"
```

### 4.4 gRPC Protocol Buffers定義

**ファイル**: `proto/embedding.proto`

```protobuf
syntax = "proto3";

package embedding;

option go_package = "github.com/yourorg/code-rag-indexer/proto/embedding";

// 埋め込みサービス
service EmbeddingService {
  // 単一コードのベクトル化
  rpc EmbedSingle(EmbedRequest) returns (EmbedResponse);
  
  // バッチコードのベクトル化
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
  
  // ヘルスチェック
  rpc Health(HealthRequest) returns (HealthResponse);
}

// 単一埋め込みリクエスト
message EmbedRequest {
  string text = 1;              // コードテキスト
  int32 max_length = 2;         // 最大トークン長（オプション）
}

// 単一埋め込みレスポンス
message EmbedResponse {
  repeated float vector = 1;    // 768次元ベクトル
  int32 tokens_used = 2;        // 使用トークン数
}

// バッチ埋め込みリクエスト
message EmbedBatchRequest {
  repeated string texts = 1;    // コードテキストのリスト
  int32 max_length = 2;         // 最大トークン長（オプション）
}

// バッチ埋め込みレスポンス
message EmbedBatchResponse {
  repeated Vector vectors = 1;  // ベクトルのリスト
}

message Vector {
  repeated float values = 1;    // 768次元ベクトル
  int32 tokens_used = 2;        // 使用トークン数
}

// ヘルスチェックリクエスト
message HealthRequest {}

// ヘルスチェックレスポンス
message HealthResponse {
  bool healthy = 1;             // サーバー状態
  string model_name = 2;        // ロード済みモデル名
  string device = 3;            // 使用デバイス（cuda/cpu）
}
```

### 4.5 .ragignore ファイル仕様

**.gitignore互換の文法**

```
# コメント
*.pyc
__pycache__/
build/
dist/
*.test.js
*_test.go
generated/
```

### 4.6 ログ出力仕様

#### Goメインプログラムのログフォーマット

```
[TIMESTAMP] [LEVEL] [COMPONENT] MESSAGE
```

**例**:
```
[2025-11-23 10:30:45] [INFO] [Parser] Starting to parse file: src/main.py
[2025-11-23 10:30:46] [WARN] [Parser] Skipped file due to syntax error: src/broken.py
[2025-11-23 10:30:50] [INFO] [Embedder] Sent batch request: 8 codes
[2025-11-23 10:30:51] [INFO] [Indexer] Registered 150 functions to collection 'my-project'
[2025-11-23 10:30:51] [ERROR] [EmbeddingClient] gRPC connection failed: connection refused
```

#### Python埋め込みサーバーのログフォーマット

```
[TIMESTAMP] [LEVEL] [gRPC] MESSAGE
```

**例**:
```
[2025-11-23 10:30:00] [INFO] [gRPC] Server started on localhost:50051
[2025-11-23 10:30:05] [INFO] [gRPC] Model loaded: jinaai/jina-embeddings-v2-base-code
[2025-11-23 10:30:50] [INFO] [gRPC] EmbedBatch request: 8 texts
[2025-11-23 10:30:51] [INFO] [gRPC] EmbedBatch response: 8 vectors (768 dim)
[2025-11-23 10:31:00] [ERROR] [gRPC] Invalid request: text is empty
```

#### ログレベル

- **DEBUG**: 詳細なデバッグ情報
- **INFO**: ファイル処理開始、登録完了などの通常情報
- **WARN**: 構文エラー、スキップしたファイルなどの警告
- **ERROR**: 接続エラー、致命的な問題

---

## 5. 制約事項

### 5.1 技術的制約

| 制約ID | 制約内容 |
|--------|---------|
| CNS-001 | 埋め込みモデルはPython埋め込みサーバーで実行される |
| CNS-002 | gRPC通信はローカルネットワーク内に限定される（TLS未使用） |
| CNS-003 | Qdrantバージョン 1.7.0以降が必要 |
| CNS-004 | Go 1.21以上が必要 |
| CNS-005 | Python 3.11以上が必要（埋め込みサーバー） |
| CNS-006 | ファイルエンコーディングはUTF-8を推奨 |
| CNS-007 | MCPサーバーと同一の埋め込みモデル・バージョンを使用する必要がある |

### 5.2 運用上の制約

| 制約ID | 制約内容 |
|--------|---------|
| CNS-101 | Python埋め込みサーバーは事前に起動しておく必要がある |
| CNS-102 | Qdrantサーバーは事前に起動しておく必要がある |
| CNS-103 | 埋め込みモデルは初回実行時に自動ダウンロードされる（数GB） |
| CNS-104 | コレクションの上書き時、既存データは完全に削除される |
| CNS-105 | モデルを変更した場合、既存コレクションの再登録が必要 |
| CNS-106 | gRPC通信のタイムアウトはデフォルト30秒 |

---

## 6. 配布・デプロイメント要件

### 6.1 配布形態

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| DEP-001 | Goメインプログラムをシングルバイナリで配布すること | 必須 | クロスコンパイル対応 |
| DEP-002 | Python埋め込みサーバーをDockerイメージで配布すること | 必須 | 依存関係を含む |
| DEP-003 | Docker Composeでの一括起動に対応すること | 必須 | Qdrant + 埋め込みサーバー + Indexer |
| DEP-004 | systemdサービスファイルを提供すること | 推奨 | 埋め込みサーバーの常駐化 |

### 6.2 Goの依存パッケージ

```go
// go.mod
module github.com/yourorg/code-rag-indexer

go 1.21

require (
    github.com/smacker/go-tree-sitter v0.0.0-20231219031718-233c2616a01b
    github.com/qdrant/go-client v1.7.0
    google.golang.org/grpc v1.60.0
    google.golang.org/protobuf v1.31.0
    gopkg.in/yaml.v3 v3.0.1
    github.com/sirupsen/logrus v1.9.3
    github.com/spf13/cobra v1.8.0  // CLI
    github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00  // .ragignore
)
```

### 6.3 Pythonの依存パッケージ

```txt
# requirements.txt（埋め込みサーバー）
grpcio>=1.60.0
grpcio-tools>=1.60.0
transformers>=4.35.0
torch>=2.2.0
```

### 6.4 Docker構成

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Qdrantサーバー
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage

  # Python埋め込みサーバー
  embedding-server:
    build:
      context: ./embedding-server
      dockerfile: Dockerfile
    ports:
      - "50051:50051"
    volumes:
      - ./model_cache:/root/.cache/huggingface
    environment:
      - LOG_LEVEL=INFO
    command: ["--host", "0.0.0.0", "--port", "50051"]

  # Goインデクサー（ワンショット実行）
  indexer:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./source_code:/source_code:ro
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    depends_on:
      - qdrant
      - embedding-server
    command: ["-c", "/app/config.yaml"]
```

---

## 7. 将来拡張

### Phase 2 機能（優先度：中）

| 機能ID | 機能内容 | 実装時期目安 |
|--------|---------|------------|
| FUT-001 | Gitリポジトリクローン機能 | Phase 2 |
| FUT-002 | 周辺コンテキストのベクトル化 | Phase 2 |
| FUT-003 | 差分更新機能（変更ファイルのみ再登録） | Phase 2 |
| FUT-004 | 被呼び出し関係の完全な解析 | Phase 2 |
| FUT-005 | gRPC通信のTLS対応 | Phase 2 |
| FUT-006 | 埋め込みサーバーの動的スケーリング | Phase 2 |

### Phase 3 機能（優先度：低）

| 機能ID | 機能内容 | 実装時期目安 |
|--------|---------|------------|
| FUT-101 | Web UIでの進捗確認 | Phase 3 |
| FUT-102 | 複数リポジトリの一括処理 | Phase 3 |
| FUT-103 | カスタムパーサープラグイン機構 | Phase 3 |
| FUT-104 | メトリクス収集・可視化（Prometheus） | Phase 3 |
| FUT-105 | 埋め込みサーバーのロードバランシング | Phase 3 |

---

## 8. 用語集

| 用語 | 定義 |
|------|------|
| AST | Abstract Syntax Tree（抽象構文木）。ソースコードの構造を木構造で表現したもの |
| RAG | Retrieval-Augmented Generation。検索拡張生成。外部知識を検索してLLMの回答を補強する技術 |
| ベクトル化 | テキストを数値ベクトルに変換する処理。埋め込み（Embedding）とも呼ばれる |
| Qdrant | オープンソースのベクトルデータベース |
| コレクション | Qdrantにおけるデータの論理的なグループ。リポジトリごとに作成 |
| チャンク | RAGにおける検索単位。本ツールでは関数・メソッド単位 |
| Jina Embeddings | Jina AIが提供するコード特化型の埋め込みモデル。8192トークンの長いコンテキストに対応 |
| MCP | Model Context Protocol。LLMと外部ツール・データソースを連携させるプロトコル |
| Tree-sitter | 複数のプログラミング言語に対応した高速なパーサーライブラリ |
| gRPC | Google Remote Procedure Call。高性能なRPCフレームワーク |
| Protocol Buffers | gRPCで使用されるインターフェース定義言語（IDL）とシリアライゼーション形式 |

---

## 9. 参考資料

- IPA「ユーザのための要件定義ガイド」
- Qdrant公式ドキュメント: https://qdrant.tech/documentation/
- Jina Embeddings v2 Code: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- gRPC公式ドキュメント: https://grpc.io/docs/
- Protocol Buffers: https://protobuf.dev/

---

## 10. 変更履歴

**v1.0 → v2.0 の主な変更点:**

1. **実装言語の変更**
   - 変更前: Python 3.11以上
   - 変更後: Go 1.21以上（メインプログラム）+ Python 3.11以上（埋め込みサーバー）
   - 理由: シングルバイナリ配布、高速な並列処理、型安全性の向上

2. **アーキテクチャの変更**
   - 変更前: Python単体
   - 変更後: Go（AST解析・Qdrant登録）+ Python（埋め込みサーバー）
   - 通信方式: gRPC（HTTP/RESTではなく）

3. **新規要件の追加**
   - VEC-001～VEC-007: gRPC通信関連要件
   - EMB-001～EMB-008: 埋め込みサーバー機能要件
   - EMB-101～EMB-107: gRPCクライアント機能要件
   - CFG-004: 埋め込みサーバー接続設定
   - DEP-004: systemdサービスファイル

4. **インターフェース定義の追加**
   - Protocol Buffers定義（embedding.proto）
   - 埋め込みサーバー起動コマンド
   - gRPCアドレス設定

5. **配布形態の明確化**
   - Goバイナリ: シングルバイナリ（クロスコンパイル）
   - Pythonサーバー: Dockerイメージ
   - Docker Compose: 3サービス構成

---

**文書履歴**

| バージョン | 日付 | 変更内容 | 作成者 |
|-----------|------|---------|--------|
| 1.0 | 2025-11-23 | 初版作成（Python版） | - |
| 1.1 | 2025-11-23 | モデル変更、配布形態変更 | - |
| 2.0 | 2025-11-23 | Go言語版への変更、gRPC埋め込みサーバー追加 | - |