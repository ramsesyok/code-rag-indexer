# ソースコードRAG登録ツール + MCPサーバー 統合要件定義書

**文書バージョン**: 3.0  
**作成日**: 2025-11-23  
**更新日**: 2025-11-24（MCPサーバー統合）  
**プロジェクト名**: Code RAG System (Indexer + MCP Server)

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0 | 2025-11-23 | 初版作成（Python版） |
| 2.0 | 2025-11-23 | Go言語版への変更、gRPC埋め込みサーバー追加 |
| 3.0 | 2025-11-24 | MCPサーバーを統合、Python実装として追加 |

---

## 1. プロジェクトの概要

### 1.1 背景と目的

本プロジェクトは、ローカルLLMとローカルRAGを活用したソースコード理解支援システムの**完全なエコシステム**を構築する。大規模なコードベース（10万行～100万行超）の理解、リファクタリング支援、バグ調査を効率化するため、以下の3つのコンポーネントを統合する：

1. **Indexer（Go）**: ソースコードをAST解析してQdrantに登録
2. **Embedding Server（Python）**: コード特化型埋め込みモデルのgRPCサーバー
3. **MCP Server（Python）**: LLMから呼び出し可能な検索インターフェース

### 1.2 システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop / LLM                     │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Protocol
                       v
┌─────────────────────────────────────────────────────────────┐
│               MCP Server (Python)                           │
│  - search_code tool                                         │
│  - get_function_details tool                                │
│  - list_collections tool                                    │
└─────────────┬────────────────────────┬──────────────────────┘
              │ gRPC                   │ HTTP
              v                        v
┌─────────────────────────┐   ┌──────────────────────────────┐
│  Embedding Server       │   │  Qdrant Vector Database      │
│  (Python, gRPC)         │   │  - Collections               │
│  - EmbedBatch           │   │  - Vectors (768 dim)         │
│  - ListModels           │   │  - Metadata                  │
└─────────────────────────┘   └─────────────┬────────────────┘
                                            ↑
                                            │ HTTP
┌───────────────────────────────────────────┼────────────────┐
│  Indexer (Go)                             │                │
│  - File Scanner                           │                │
│  - AST Parser (Tree-sitter)               │                │
│  - Metadata Extraction                    │                │
│  - Parallel Processing                    │                │
└───────────────────────────────────────────┴────────────────┘
```

### 1.3 適用範囲

- **対象ユーザー**: ソフトウェア開発チーム
- **対象コード**: C、C++、Java、Rust、Go、Python で記述されたソースコード
- **利用環境**: 
  - Indexer: Windows、Linux、macOS（シングルバイナリ）
  - Embedding Server: Docker
  - MCP Server: Docker
  - Qdrant: Docker
- **実装言語**: 
  - Indexer: Go 1.21以上
  - Embedding Server: Python 3.11以上
  - MCP Server: Python 3.11以上

---

## 2. システム全体の機能要件

### 2.1 コンポーネント間の関係

| コンポーネント | 役割 | 入力 | 出力 |
|--------------|------|------|------|
| **Indexer** | ソースコード登録 | ローカルディレクトリ | Qdrantコレクション |
| **Embedding Server** | ベクトル化 | テキスト（gRPC） | ベクトル（768次元） |
| **MCP Server** | 検索・取得 | クエリ（MCP） | 検索結果（JSON） |
| **Qdrant** | ベクトル保存・検索 | ベクトル+メタデータ | 検索結果 |

### 2.2 データフロー

#### 2.2.1 登録フロー

```
[ソースコード]
    ↓ (1) File Scanner
[ファイルリスト]
    ↓ (2) AST Parser
[FunctionInfo配列]
    ↓ (3) Embedding Server (gRPC)
[Vector配列]
    ↓ (4) Qdrant Indexer
[Qdrantコレクション]
```

#### 2.2.2 検索フロー

```
[LLM/Claude]
    ↓ (1) MCP Tool Call
[MCP Server]
    ↓ (2) Embedding Server (gRPC)
[クエリベクトル]
    ↓ (3) Qdrant Search
[検索結果]
    ↓ (4) FunctionInfo復元
[MCP Server]
    ↓ (5) MCP Response
[LLM/Claude]
```

---

## 3. Indexer機能要件（Go）

### 3.1 入力機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-IN-001 | ローカルディレクトリを入力として受け付けること | 必須 | 初期実装対象 |
| IDX-IN-002 | Gitリポジトリをクローンして入力として受け付けること | 将来 | Phase 2で実装 |
| IDX-IN-003 | 設定ファイル（YAML形式）から入力パスを読み込めること | 必須 | |
| IDX-IN-101 | .ragignoreファイルで除外パターンを指定できること | 必須 | .gitignore互換の文法 |
| IDX-IN-102 | バイナリファイルを自動検出して除外すること | 必須 | |
| IDX-IN-103 | 対象外の拡張子を設定で指定できること | 必須 | テストコード、自動生成コードの除外に使用 |

### 3.2 AST構造分析機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-AST-001 | 対象言語（C/C++/Java/Rust/Go/Python）のソースコードをAST解析すること | 必須 | Tree-sitter使用 |
| IDX-AST-002 | 関数・メソッド単位でコードを分割すること | 必須 | RAGのチャンク単位 |
| IDX-AST-003 | 構文エラーのあるファイルは警告を出力してスキップすること | 必須 | パース不可能なファイル |
| IDX-AST-004 | 意味的に不完全なコードは警告を出力して登録すること | 必須 | 開発途中のコードも対象 |

#### 3.2.1 抽出情報

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-AST-101 | 関数名、引数、戻り値の型を抽出すること | 必須 | |
| IDX-AST-102 | クラス階層、継承関係を抽出すること | 必須 | |
| IDX-AST-103 | 依存関係（import/include）を抽出すること | 必須 | |
| IDX-AST-104 | コメント、ドキュメント文字列を抽出すること | 必須 | |
| IDX-AST-105 | スコープ情報（グローバル/クラスメンバー/ローカル）を抽出すること | 必須 | |
| IDX-AST-106 | 修飾子（public/private/protected/static/const/async等）を抽出すること | 必須 | |
| IDX-AST-107 | 位置情報（開始行・終了行・開始列・終了列）を抽出すること | 必須 | エラー箇所特定に使用 |
| IDX-AST-108 | 呼び出し関係（この関数が呼び出す他の関数）を抽出すること | 必須 | |
| IDX-AST-109 | 被呼び出し関係（この関数を呼び出す関数）を抽出すること | 推奨 | 逆引き用、要追加解析 |
| IDX-AST-110 | 使用している型の一覧を抽出すること | 必須 | |
| IDX-AST-111 | サイクロマティック複雑度を計算すること | 推奨 | バグ潜在箇所の特定に有用 |
| IDX-AST-112 | 実効行数、コメント行数を計算すること | 推奨 | |
| IDX-AST-113 | 言語固有の属性（デコレータ/アノテーション/属性等）を抽出すること | 推奨 | |

### 3.3 ベクトル化・登録機能

#### 3.3.1 ベクトル化（gRPC経由）

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-VEC-001 | Python埋め込みサーバーとgRPCで通信すること | 必須 | HTTP/RESTではなくgRPC |
| IDX-VEC-002 | jinaai/jina-embeddings-v2-base-codeモデル（768次元）を使用してコードをベクトル化すること | 必須 | サーバー側で実行 |
| IDX-VEC-003 | 最大8192トークンまでのコンテキストを処理できること | 必須 | 長い関数・クラスに対応 |
| IDX-VEC-004 | コード本体とコメントを結合してベクトル化すること | 必須 | |
| IDX-VEC-005 | バッチリクエストに対応すること | 必須 | 複数コードを一度に送信 |
| IDX-VEC-006 | gRPC接続エラー時にリトライすること | 必須 | 3回までリトライ |
| IDX-VEC-007 | 埋め込みサーバーのヘルスチェックを実行できること | 必須 | 起動前の確認 |
| IDX-VEC-008 | MCPサーバーと同一のモデル・バージョンを使用すること | 必須 | ベクトル空間の一致が必要 |

#### 3.3.2 Qdrantへの登録

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-QDR-001 | Qdrantに関数・メソッド単位でベクトルを登録すること | 必須 | |
| IDX-QDR-002 | AST解析で抽出した全メタ情報をペイロードとして保存すること | 必須 | docs/data-schema.md参照 |
| IDX-QDR-003 | ペイロードはMCPサーバーと共通のスキーマに従うこと | 必須 | 一貫性保証 |

### 3.4 コレクション管理機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| IDX-COL-001 | コレクション名をカスタム指定できること | 必須 | 設定ファイルで指定 |
| IDX-COL-002 | コレクション名をディレクトリ名ベースで自動生成できること | 必須 | デフォルト動作 |
| IDX-COL-003 | リポジトリ名ベースでコレクション名を自動生成できること | 将来 | Git対応時に実装 |
| IDX-COL-004 | 既存コレクションが存在する場合、削除して再作成すること | 必須 | 上書き動作 |
| IDX-COL-005 | リポジトリごとに独立したコレクションを作成すること | 必須 | |

### 3.5 性能要件

| 要件ID | 要件内容 | 目標値 | 備考 |
|--------|---------|--------|------|
| IDX-PRF-001 | ファイル単位で並列処理を行うこと | - | Goroutineで並列化 |
| IDX-PRF-002 | 16GB～32GBのメモリで動作すること | 32GB以下 | 一般的なPC環境 |
| IDX-PRF-003 | 100万行規模のコードベースを一晩（8時間以内）で処理できること | 8時間以内 | PoCレベル目標 |

---

## 4. Embedding Server機能要件（Python）

### 4.1 サーバー側機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| EMB-001 | gRPCサーバーとして起動すること | 必須 | Python実装 |
| EMB-002 | Jina Embeddings v2 Codeモデルを読み込むこと | 必須 | 起動時に1回 |
| EMB-003 | バッチコードのベクトル化リクエストに対応すること | 必須 | EmbedBatch RPC |
| EMB-004 | ヘルスチェックリクエストに対応すること | 必須 | Health RPC |
| EMB-005 | サポートモデル一覧リクエストに対応すること | 必須 | ListModels RPC |
| EMB-006 | GPU/CPU自動切り替えに対応すること | 必須 | CUDA利用可能時はGPU |
| EMB-007 | エラー時は適切なgRPCステータスコードを返すこと | 必須 | INVALID_ARGUMENT等 |
| EMB-008 | リクエストログを出力すること | 推奨 | デバッグ用 |

### 4.2 モデル管理機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| EMB-101 | 複数モデルに対応すること | 必須 | リクエストにモデル名指定 |
| EMB-102 | モデルを遅延ロード（初回アクセス時のみ）すること | 必須 | メモリ効率化 |
| EMB-103 | ロード済みモデルをキャッシュすること | 必須 | 2回目以降高速化 |
| EMB-104 | デフォルトモデルを設定できること | 必須 | モデル名未指定時に使用 |
| EMB-105 | 起動時に指定モデルをプリロードできること | 推奨 | 初回レスポンス高速化 |

### 4.3 対応モデル

| モデル名 | 次元数 | 最大トークン | 優先度 |
|---------|--------|-------------|--------|
| jinaai/jina-embeddings-v2-base-code | 768 | 8192 | 必須（デフォルト） |
| microsoft/unixcoder-base | 768 | 512 | 推奨 |
| bigcode/starencoder | 768 | 2048 | 推奨 |

---

## 5. MCP Server機能要件（Python）

### 5.1 MCP Protocol対応

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-001 | MCP Protocolに準拠すること | 必須 | JSON-RPC over stdio |
| MCP-002 | Claude DesktopまたはCLIから呼び出し可能なこと | 必須 | |
| MCP-003 | 複数ツールを提供すること | 必須 | 最低3ツール |
| MCP-004 | エラー時は適切なMCPエラーレスポンスを返すこと | 必須 | |

### 5.2 提供ツール

#### 5.2.1 search_code ツール

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-SEARCH-001 | 自然言語クエリでコードを検索できること | 必須 | ベクトル検索 |
| MCP-SEARCH-002 | 検索結果を関連度順にソートして返すこと | 必須 | スコアリング |
| MCP-SEARCH-003 | 検索件数を指定できること | 必須 | デフォルト10件 |
| MCP-SEARCH-004 | プログラミング言語でフィルタできること | 推奨 | metadata filter |
| MCP-SEARCH-005 | 複雑度範囲でフィルタできること | 推奨 | metadata filter |
| MCP-SEARCH-006 | ファイルパスパターンでフィルタできること | 推奨 | metadata filter |
| MCP-SEARCH-007 | コレクションを指定して検索できること | 必須 | 複数リポジトリ対応 |

**入力スキーマ**:
```json
{
  "query": "ユーザー認証を処理する関数",
  "collection": "my-project",
  "limit": 10,
  "filters": {
    "language": "python",
    "min_complexity": 1,
    "max_complexity": 10,
    "file_pattern": "*/auth/*"
  }
}
```

**出力スキーマ**:
```json
{
  "results": [
    {
      "function_name": "authenticate_user",
      "file_path": "/src/auth/handlers.py",
      "start_line": 45,
      "end_line": 67,
      "language": "python",
      "code": "def authenticate_user(username, password): ...",
      "docstring": "Authenticate user with credentials",
      "score": 0.95
    }
  ],
  "total": 1
}
```

#### 5.2.2 get_function_details ツール

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-DETAIL-001 | 関数の詳細情報を取得できること | 必須 | 全メタデータ |
| MCP-DETAIL-002 | ファイルパスと行番号で関数を特定できること | 必須 | |
| MCP-DETAIL-003 | 関数の依存関係を含めて返すこと | 推奨 | imports, calls |
| MCP-DETAIL-004 | 呼び出し元の関数リストを返すこと | 将来 | 逆引き検索 |

**入力スキーマ**:
```json
{
  "collection": "my-project",
  "file_path": "/src/auth/handlers.py",
  "start_line": 45
}
```

**出力スキーマ**:
```json
{
  "function_name": "authenticate_user",
  "file_path": "/src/auth/handlers.py",
  "start_line": 45,
  "end_line": 67,
  "language": "python",
  "function_type": "function",
  "code": "def authenticate_user(username, password): ...",
  "arguments": ["username", "password"],
  "return_type": "bool",
  "docstring": "Authenticate user with credentials",
  "imports": ["hashlib", "jwt"],
  "calls": ["hash_password", "verify_token"],
  "modifiers": [],
  "scope": "global",
  "loc": 23,
  "complexity": 5
}
```

#### 5.2.3 list_collections ツール

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-LIST-001 | Qdrantに登録されているコレクション一覧を取得できること | 必須 | |
| MCP-LIST-002 | 各コレクションの統計情報を含めること | 推奨 | 件数、言語分布等 |

**入力スキーマ**:
```json
{}
```

**出力スキーマ**:
```json
{
  "collections": [
    {
      "name": "project-a",
      "vector_count": 15234,
      "languages": ["python", "javascript"],
      "created_at": "2025-11-20T10:30:00Z"
    },
    {
      "name": "project-b",
      "vector_count": 8932,
      "languages": ["rust", "go"],
      "created_at": "2025-11-22T15:45:00Z"
    }
  ]
}
```

### 5.3 検索機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-SRCH-001 | クエリをベクトル化すること | 必須 | Embedding Server経由 |
| MCP-SRCH-002 | Qdrantでベクトル検索を実行すること | 必須 | コサイン類似度 |
| MCP-SRCH-003 | メタデータフィルタを適用すること | 必須 | 言語、複雑度等 |
| MCP-SRCH-004 | 検索結果をFunctionInfoに復元すること | 必須 | ペイロードから復元 |
| MCP-SRCH-005 | スコアリングとランキングを行うこと | 必須 | |
| MCP-SRCH-006 | 検索タイムアウトを設定できること | 必須 | デフォルト30秒 |

### 5.4 統合機能

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| MCP-INT-001 | Embedding Serverと同じモデルを使用すること | 必須 | ベクトル空間の一致 |
| MCP-INT-002 | Indexerと同じペイロードスキーマを使用すること | 必須 | データ互換性 |
| MCP-INT-003 | gRPCクライアントでEmbedding Serverに接続すること | 必須 | |
| MCP-INT-004 | HTTP/gRPCクライアントでQdrantに接続すること | 必須 | |

### 5.5 性能要件

| 要件ID | 要件内容 | 目標値 | 備考 |
|--------|---------|--------|------|
| MCP-PRF-001 | 検索レスポンスタイムが1秒以内であること | 1秒以内 | P95 |
| MCP-PRF-002 | 10 req/secの同時検索に対応すること | 10 req/sec | 対話的利用 |
| MCP-PRF-003 | メモリ使用量が200MB以内であること | 200MB以内 | Docker環境 |

---

## 6. データスキーマ要件

### 6.1 Qdrant Payload Schema

**管理方法**: `docs/data-schema.md` に詳細定義を記載

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| SCHEMA-001 | Indexer（Go）とMCP Server（Python）で同一スキーマを使用すること | 必須 | |
| SCHEMA-002 | スキーマドキュメントを真実の情報源（Single Source of Truth）とすること | 必須 | |
| SCHEMA-003 | スキーマ変更時はドキュメント→実装の順で更新すること | 必須 | |
| SCHEMA-004 | 全フィールドの型、必須/任意、説明を定義すること | 必須 | |

### 6.2 FunctionInfo スキーマ

**必須フィールド**:

| フィールド名 | 型 | 説明 | Goフィールド | Python フィールド |
|------------|-----|------|-------------|------------------|
| function_name | string | 関数名 | Name | function_name |
| file_path | string | ファイルパス | FilePath | file_path |
| start_line | int | 開始行 | StartLine | start_line |
| end_line | int | 終了行 | EndLine | end_line |
| language | string | 言語 | Language | language |
| function_type | string | 種別 | FunctionType | function_type |
| code | string | コード本体 | Code | code |
| scope | string | スコープ | Scope | scope |
| loc | int | 行数 | LOC | loc |

**オプションフィールド**:

| フィールド名 | 型 | 説明 | Goフィールド | Python フィールド |
|------------|-----|------|-------------|------------------|
| arguments | string[] | 引数リスト | Arguments | arguments |
| return_type | string | 戻り値型 | ReturnType | return_type |
| docstring | string | ドキュメント | Docstring | docstring |
| comments | string[] | コメント | Comments | comments |
| modifiers | string[] | 修飾子 | Modifiers | modifiers |
| imports | string[] | インポート | Imports | imports |
| calls | string[] | 呼び出し | Calls | calls |
| complexity | int | 複雑度 | Complexity | complexity |
| comment_lines | int | コメント行数 | CommentLines | comment_lines |

### 6.3 スキーマバージョニング

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| SCHEMA-VER-001 | スキーマにバージョン番号を付与すること | 推奨 | 将来の互換性 |
| SCHEMA-VER-002 | 破壊的変更時はメジャーバージョンを上げること | 推奨 | |
| SCHEMA-VER-003 | フィールド追加時はマイナーバージョンを上げること | 推奨 | |

---

## 7. Protocol Buffers 要件（gRPC）

### 7.1 Embedding Service

**ファイル**: `proto/embedding.proto`

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| PROTO-001 | EmbedBatch RPCを定義すること | 必須 | バッチベクトル化 |
| PROTO-002 | Health RPCを定義すること | 必須 | ヘルスチェック |
| PROTO-003 | ListModels RPCを定義すること | 必須 | モデル一覧取得 |
| PROTO-004 | Go、Python両方のコード生成に対応すること | 必須 | |
| PROTO-005 | リクエストにmodel_nameフィールドを含めること | 必須 | 複数モデル対応 |

### 7.2 コード生成

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| PROTO-GEN-001 | Makefileでコード生成を自動化すること | 必須 | make proto |
| PROTO-GEN-002 | Go側は proto/embedding/ に生成すること | 必須 | |
| PROTO-GEN-003 | Python側（Embedding Server）は embedding-server/proto/embedding/ に生成すること | 必須 | |
| PROTO-GEN-004 | Python側（MCP Server）は mcp-server/proto/embedding/ に生成すること | 必須 | |

---

## 8. 非機能要件

### 8.1 可用性・信頼性要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| REL-001 | Indexer: 一部ファイルの解析失敗時も、処理を継続すること | 警告を出力してスキップ |
| REL-002 | Indexer: 処理中断時、どのファイルまで処理したか記録すること | ログから追跡可能にする |
| REL-003 | Indexer: 埋め込みサーバーダウン時はエラー終了すること | 自動復旧は行わない |
| REL-004 | Indexer: Qdrant接続断時はリトライすること | 最大3回 |
| REL-005 | Embedding Server: gRPCサーバーが起動失敗時はエラー表示すること | ポート競合等 |
| REL-006 | MCP Server: Embedding Server接続失敗時はエラー終了すること | |
| REL-007 | MCP Server: Qdrant接続失敗時はエラー終了すること | |
| REL-008 | MCP Server: 検索タイムアウト時は適切なエラーを返すこと | |

### 8.2 保守性・拡張性要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| MNT-001 | 新しいプログラミング言語の追加が容易な設計とすること | パーサーの追加のみで対応 |
| MNT-002 | 埋め込みモデルの変更が容易な設計とすること | サーバー側の設定変更のみ |
| MNT-003 | Indexerはシングルバイナリで配布できること | Go単体でコンパイル可能 |
| MNT-004 | MCPサーバーとIndexerでモデル設定を共有できること | 設定ファイルの共通化 |
| MNT-005 | 埋め込みサーバーを独立して起動・停止できること | サービス化対応 |
| MNT-006 | スキーマドキュメントで一貫性を保つこと | docs/data-schema.md |
| MNT-007 | 新しいMCPツールの追加が容易な設計とすること | プラグイン方式 |

### 8.3 移植性要件

| 要件ID | 要件内容 | 対象環境 |
|--------|---------|---------|
| PRT-001 | IndexerがWindows環境で動作すること | Windows 10以降 |
| PRT-002 | IndexerがLinux環境で動作すること | Ubuntu 20.04以降、RHEL 8以降 |
| PRT-003 | IndexerがmacOS環境で動作すること | macOS 11以降 |
| PRT-004 | Indexerが Go 1.21以上で動作すること | 型パラメータ等の機能を活用 |
| PRT-005 | Embedding ServerがPython 3.11以上で動作すること | |
| PRT-006 | MCP ServerがPython 3.11以上で動作すること | |
| PRT-007 | Docker Composeで全環境を起動できること | 開発環境 |

### 8.4 セキュリティ要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| SEC-001 | Qdrant接続時、APIキー認証を使用すること | 設定ファイルから読み込み |
| SEC-002 | APIキーを環境変数から読み込めること | 設定ファイルへの平文保存を回避 |
| SEC-003 | gRPC通信はローカル環境に限定すること | TLS不要 |
| SEC-004 | MCP ServerはClaude Desktopからのみアクセス可能とすること | stdio経由 |
| SEC-005 | 機密情報のフィルタリングは行わないこと | ユーザー責任で管理 |

### 8.5 運用要件

| 要件ID | 要件内容 | 備考 |
|--------|---------|------|
| OPR-001 | IndexerはCLIインターフェースを提供すること | GUI不要 |
| OPR-002 | 全コンポーネントがログをファイルとコンソールに出力すること | 両方に同時出力 |
| OPR-003 | Indexerがファイル単位の読み込み開始を通知すること | 進捗確認用 |
| OPR-004 | 警告・エラーを明確に区別してログ出力すること | |
| OPR-005 | Indexer処理完了時、登録件数のサマリーを表示すること | |
| OPR-006 | Embedding Serverを別プロセスで起動すること | systemdやsupervisor推奨 |
| OPR-007 | MCP Serverを別プロセスで起動すること | Claude Desktop設定 |
| OPR-008 | Docker Composeで全サービスを一括起動できること | |

---

## 9. インターフェース要件

### 9.1 Indexer CLI

```bash
code-rag-indexer [OPTIONS]
```

| オプション | 説明 | 必須/任意 |
|----------|------|---------|
| `-c, --config <PATH>` | 設定ファイルのパス | 必須 |
| `-v, --verbose` | 詳細ログ出力 | 任意 |
| `-h, --help` | ヘルプ表示 | 任意 |
| `--version` | バージョン表示 | 任意 |

### 9.2 Embedding Server CLI

```bash
embedding-server [OPTIONS]
```

| オプション | 説明 | 必須/任意 | デフォルト |
|----------|------|---------|-----------|
| `--host <HOST>` | バインドするホスト | 任意 | localhost |
| `--port <PORT>` | バインドするポート | 任意 | 50051 |
| `--config <PATH>` | 設定ファイルパス | 任意 | embedding-server.yaml |

### 9.3 MCP Server 設定（Claude Desktop）

**ファイル**: `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "code-rag": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--network", "host",
        "mcp-server:latest"
      ]
    }
  }
}
```

### 9.4 設定ファイル仕様

#### 9.4.1 Indexer設定（config.yaml）

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

# 埋め込みサーバー設定
embedding_server:
  address: "localhost:50051"
  timeout: 30s
  max_retries: 3
  batch_size: 8
  model_name: "jinaai/jina-embeddings-v2-base-code"

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

#### 9.4.2 Embedding Server設定（embedding-server.yaml）

```yaml
# サーバー設定
server:
  host: "0.0.0.0"
  port: 50051
  max_workers: 10

# デフォルトモデル
default_model: "jinaai/jina-embeddings-v2-base-code"

# サポートするモデル一覧
models:
  - name: "jinaai/jina-embeddings-v2-base-code"
    dimension: 768
    max_length: 8192
    trust_remote_code: true
    preload: true
  
  - name: "microsoft/unixcoder-base"
    dimension: 768
    max_length: 512
    trust_remote_code: false
    preload: false

# デバイス設定
device: "auto"  # auto/cuda/cpu

# ログ設定
logging_level: "INFO"
logging_file: "/var/log/embedding-server.log"
```

#### 9.4.3 MCP Server設定（mcp-server.yaml）

```yaml
# Qdrant設定
qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"

# 埋め込みサーバー設定
embedding_server:
  address: "localhost:50051"
  timeout: 30s
  max_retries: 3
  model_name: "jinaai/jina-embeddings-v2-base-code"

# 検索設定
search:
  default_limit: 10
  max_limit: 50
  timeout: 30s

# ログ設定
logging:
  level: "INFO"
  file: "/var/log/mcp-server.log"
```

---

## 10. 配布・デプロイメント要件

### 10.1 配布形態

| コンポーネント | 配布形態 | 対象環境 |
|--------------|---------|---------|
| **Indexer** | シングルバイナリ | Windows/Linux/macOS |
| **Embedding Server** | Dockerイメージ | Docker環境 |
| **MCP Server** | Dockerイメージ | Docker環境 |
| **Qdrant** | Dockerイメージ（公式） | Docker環境 |

### 10.2 デプロイメント構成

| 要件ID | 要件内容 | 優先度 | 備考 |
|--------|---------|--------|------|
| DEP-001 | Goインデクサーをクロスコンパイルできること | 必須 | Windows/Linux/macOS |
| DEP-002 | Python埋め込みサーバーをDockerイメージで配布すること | 必須 | |
| DEP-003 | Python MCPサーバーをDockerイメージで配布すること | 必須 | |
| DEP-004 | Docker Composeでの一括起動に対応すること | 必須 | Qdrant + Embedding + MCP |
| DEP-005 | systemdサービスファイルを提供すること | 推奨 | 埋め込みサーバーの常駐化 |

### 10.3 docker-compose.yml 要件

```yaml
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
    build: ./embedding-server
    ports:
      - "50051:50051"
    volumes:
      - ./model_cache:/root/.cache/huggingface
      - ./embedding-server.yaml:/app/config.yaml:ro

  # MCPサーバー
  mcp-server:
    build: ./mcp-server
    stdin_open: true
    tty: true
    volumes:
      - ./mcp-server.yaml:/app/config.yaml:ro
    environment:
      - QDRANT_URL=http://qdrant:6333
      - EMBEDDING_SERVER_URL=embedding-server:50051
    depends_on:
      - qdrant
      - embedding-server
```

---

## 11. 制約事項

### 11.1 技術的制約

| 制約ID | 制約内容 |
|--------|---------|
| CNS-001 | 埋め込みモデルはPython埋め込みサーバーで実行される |
| CNS-002 | gRPC通信はローカルネットワーク内に限定される（TLS未使用） |
| CNS-003 | MCP通信はstdio経由に限定される |
| CNS-004 | Qdrantバージョン 1.7.0以降が必要 |
| CNS-005 | Go 1.21以上が必要 |
| CNS-006 | Python 3.11以上が必要（Embedding Server、MCP Server） |
| CNS-007 | ファイルエンコーディングはUTF-8を推奨 |
| CNS-008 | MCPサーバーとIndexerで同一の埋め込みモデル・バージョンを使用する必要がある |
| CNS-009 | データスキーマはdocs/data-schema.mdをSingle Source of Truthとする |

### 11.2 運用上の制約

| 制約ID | 制約内容 |
|--------|---------|
| CNS-101 | Python埋め込みサーバーは事前に起動しておく必要がある |
| CNS-102 | Qdrantサーバーは事前に起動しておく必要がある |
| CNS-103 | 埋め込みモデルは初回実行時に自動ダウンロードされる（数GB） |
| CNS-104 | コレクションの上書き時、既存データは完全に削除される |
| CNS-105 | モデルを変更した場合、既存コレクションの再登録が必要 |
| CNS-106 | gRPC通信のタイムアウトはデフォルト30秒 |
| CNS-107 | MCP ServerはClaude Desktop経由でのみ利用可能 |

---

## 12. 将来拡張

### Phase 2 機能（優先度：中）

| 機能ID | 機能内容 | 実装時期目安 |
|--------|---------|------------|
| FUT-001 | Gitリポジトリクローン機能（Indexer） | Phase 2 |
| FUT-002 | 周辺コンテキストのベクトル化（Indexer） | Phase 2 |
| FUT-003 | 差分更新機能（変更ファイルのみ再登録、Indexer） | Phase 2 |
| FUT-004 | 被呼び出し関係の完全な解析（Indexer） | Phase 2 |
| FUT-005 | gRPC通信のTLS対応（Embedding Server） | Phase 2 |
| FUT-006 | 埋め込みサーバーの動的スケーリング | Phase 2 |
| FUT-007 | 意味的コード検索の強化（MCP Server） | Phase 2 |
| FUT-008 | コードリファクタリング提案ツール（MCP Server） | Phase 2 |

### Phase 3 機能（優先度：低）

| 機能ID | 機能内容 | 実装時期目安 |
|--------|---------|------------|
| FUT-101 | Web UIでの進捗確認（Indexer） | Phase 3 |
| FUT-102 | 複数リポジトリの一括処理（Indexer） | Phase 3 |
| FUT-103 | カスタムパーサープラグイン機構（Indexer） | Phase 3 |
| FUT-104 | メトリクス収集・可視化（Prometheus） | Phase 3 |
| FUT-105 | 埋め込みサーバーのロードバランシング | Phase 3 |
| FUT-106 | LLMへのストリーミングレスポンス（MCP Server） | Phase 3 |
| FUT-107 | コード生成支援ツール（MCP Server） | Phase 3 |

---

## 13. 用語集

| 用語 | 定義 |
|------|------|
| AST | Abstract Syntax Tree（抽象構文木）。ソースコードの構造を木構造で表現したもの |
| RAG | Retrieval-Augmented Generation。検索拡張生成。外部知識を検索してLLMの回答を補強する技術 |
| MCP | Model Context Protocol。LLMと外部ツール・データソースを連携させるプロトコル |
| ベクトル化 | テキストを数値ベクトルに変換する処理。埋め込み（Embedding）とも呼ばれる |
| Qdrant | オープンソースのベクトルデータベース |
| コレクション | Qdrantにおけるデータの論理的なグループ。リポジトリごとに作成 |
| チャンク | RAGにおける検索単位。本システムでは関数・メソッド単位 |
| Jina Embeddings | Jina AIが提供するコード特化型の埋め込みモデル。8192トークンの長いコンテキストに対応 |
| Tree-sitter | 複数のプログラミング言語に対応した高速なパーサーライブラリ |
| gRPC | Google Remote Procedure Call。高性能なRPCフレームワーク |
| Protocol Buffers | gRPCで使用されるインターフェース定義言語（IDL）とシリアライゼーション形式 |
| stdio | Standard Input/Output。MCPサーバーとClaude Desktopの通信方式 |
| FunctionInfo | 関数のメタデータを保持する構造体/データクラス |
| Single Source of Truth | 真実の情報源。本システムではdocs/data-schema.md |

---

## 14. 参考資料

- IPA「ユーザのための要件定義ガイド」
- Qdrant公式ドキュメント: https://qdrant.tech/documentation/
- Jina Embeddings v2 Code: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- gRPC公式ドキュメント: https://grpc.io/docs/
- Protocol Buffers: https://protobuf.dev/
- MCP Protocol: https://modelcontextprotocol.io/
- Anthropic MCP Documentation: https://docs.anthropic.com/mcp/

