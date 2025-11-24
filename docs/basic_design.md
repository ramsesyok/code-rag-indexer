# Code RAG System - 基本設計書 v3.0

**文書バージョン**: 3.0  
**作成日**: 2025-11-23  
**更新日**: 2025-11-24（MCPサーバー統合）  
**対象**: Claude Code による実装  
**実装言語**: Go 1.21以上（Indexer）+ Python 3.11以上（Embedding Server + MCP Server）  
**開発手法**: テスト駆動開発（TDD）

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0 | 2025-11-23 | 初版作成（Python版） |
| 2.0 | 2025-11-23 | Go言語版への変更、gRPC埋め込みサーバー追加 |
| 2.1 | 2025-11-24 | Docker化の方針修正（Goはシングルバイナリ） |
| 3.0 | 2025-11-24 | MCPサーバー統合、構造中心の記述に変更 |

---

## 目次

1. [システムアーキテクチャ](#1-システムアーキテクチャ)
2. [プロジェクト構造](#2-プロジェクト構造)
3. [データスキーマ設計](#3-データスキーマ設計)
4. [Protocol Buffers定義](#4-protocol-buffers定義)
5. [Indexer設計（Go）](#5-indexer設計go)
6. [Embedding Server設計（Python）](#6-embedding-server設計python)
7. [MCP Server設計（Python）](#7-mcp-server設計python)
8. [依存関係管理](#8-依存関係管理)
9. [ビルド・デプロイ](#9-ビルドデプロイ)
10. [設定ファイル](#10-設定ファイル)

---

## 1. システムアーキテクチャ

### 1.1 全体構成図

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop / LLM Client              │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Protocol (stdio)
                       v
┌─────────────────────────────────────────────────────────────┐
│               MCP Server (Python/Docker)                    │
│  - MCP Tools (search_code, get_details, list_collections)  │
│  - Search Service                                           │
│  - gRPC Client (Embedder)                                   │
│  - HTTP Client (Qdrant)                                     │
└─────────────────────────┬────────────┬─────────────────────┘
                          │ gRPC       │ HTTP
                          v            v
┌─────────────────────────┴──┐   ┌────┴─────────────────────┐
│  Embedding Server (Docker) │   │  Qdrant Vector Database  │
│  - gRPC Server             │   │  - Collections           │
│  - Model Manager           │   │  - Vectors (768 dim)     │
│  - Jina Embeddings v2      │   │  - Metadata              │
└─────────────────────────────┘   └──────────────────────────┘
                                               ↑
                                               │ HTTP
┌──────────────────────────────────────────────┤
│  Indexer (Go/Single Binary)                  │
│  - File Scanner                              │
│  - AST Parser (Tree-sitter)                  │
│  - gRPC Client (Embedder)                    │
│  - HTTP Client (Qdrant)                      │
└──────────────────────────────────────────────┘
```

### 1.2 コンポーネント間通信

| 送信元 | 送信先 | プロトコル | 用途 |
|--------|--------|-----------|------|
| Indexer | Embedding Server | gRPC | コードのベクトル化 |
| Indexer | Qdrant | HTTP | ベクトル登録 |
| MCP Server | Embedding Server | gRPC | クエリのベクトル化 |
| MCP Server | Qdrant | HTTP | ベクトル検索 |
| Claude Desktop | MCP Server | stdio | ツール呼び出し |

### 1.3 データフロー

#### 登録フロー
```
ソースコード → File Scanner → AST Parser → Embedding Server → Qdrant
```

#### 検索フロー
```
Claude Desktop → MCP Server → Embedding Server → Query Vector
                     ↓
                  Qdrant Search → Results → MCP Server → Claude Desktop
```

---

## 2. プロジェクト構造

### 2.1 ディレクトリ構成

```
code-rag-indexer/                    # ルートディレクトリ
├── proto/                           # 共有Protocol Buffers定義
│   └── embedding.proto
│
├── docs/                            # ドキュメント
│   ├── requirements.md              # 要件定義書
│   ├── design.md                    # 基本設計書（本文書）
│   ├── data-schema.md               # データスキーマ（Single Source of Truth）
│   └── api/
│       ├── grpc.md                  # gRPC API仕様
│       └── mcp.md                   # MCP API仕様
│
├── pkg/                             # Go共通ライブラリ
│   └── models/
│       └── function_info.go         # FunctionInfo構造体
│
├── indexer/                         # Goインデクサー
│   ├── cmd/indexer/                 # エントリーポイント
│   ├── internal/                    # 内部パッケージ
│   │   ├── config/                  # 設定管理
│   │   ├── scanner/                 # ファイルスキャン
│   │   ├── parser/                  # AST解析
│   │   ├── embedder/                # gRPCクライアント
│   │   ├── indexer/                 # Qdrant操作
│   │   └── logger/                  # ロギング
│   ├── proto/embedding/             # 生成されたprotoコード
│   └── tests/                       # テスト
│
├── embedding-server/                # Python埋め込みサーバー
│   ├── server/                      # サーバー実装
│   │   ├── main.py                  # エントリーポイント
│   │   ├── servicer.py              # gRPC Servicer
│   │   ├── model_manager.py         # モデル管理
│   │   └── config.py                # 設定管理
│   ├── proto/embedding/             # 生成されたprotoコード
│   ├── tests/                       # テスト
│   └── Dockerfile
│
├── mcp-server/                      # MCPサーバー
│   ├── server/                      # サーバー実装
│   │   ├── main.py                  # エントリーポイント
│   │   ├── config.py                # 設定管理
│   │   ├── models.py                # データモデル
│   │   ├── tools/                   # MCPツール
│   │   ├── services/                # ビジネスロジック
│   │   └── clients/                 # 外部クライアント
│   ├── proto/embedding/             # 生成されたprotoコード
│   ├── tests/                       # テスト
│   └── Dockerfile
│
├── scripts/                         # ビルド・デプロイスクリプト
├── docker-compose.yml               # 統合デプロイ
└── Makefile                         # 統合ビルド
```

### 2.2 ファイル命名規則

| コンポーネント | 規則 | 例 |
|--------------|------|-----|
| Go ソースファイル | snake_case.go | config_loader.go |
| Go テストファイル | *_test.go | config_loader_test.go |
| Python モジュール | snake_case.py | search_service.py |
| Python テストファイル | test_*.py | test_search_service.py |
| 設定ファイル | kebab-case.yaml | mcp-server.yaml |
| ドキュメント | kebab-case.md | data-schema.md |

---

## 3. データスキーマ設計

### 3.1 スキーマ管理原則

**Single Source of Truth**: `docs/data-schema.md`

| 項目 | 説明 |
|------|------|
| **管理場所** | docs/data-schema.md に全フィールド定義 |
| **更新フロー** | スキーマドキュメント更新 → Go実装 → Python実装 |
| **一貫性保証** | Indexer（Go）とMCP Server（Python）で同一構造 |
| **変更管理** | バージョン番号付与、変更履歴記録 |

### 3.2 FunctionInfo データ構造

**概要**: 関数・メソッドのメタデータを保持

#### 必須フィールド

| Field | Type | 説明 |
|-------|------|------|
| function_name | string | 関数名 |
| file_path | string | ファイルパス |
| start_line | int | 開始行（1-indexed） |
| end_line | int | 終了行（1-indexed） |
| start_column | int | 開始列（0-indexed） |
| end_column | int | 終了列（0-indexed） |
| language | string | プログラミング言語 |
| function_type | string | 種別（function/method/class） |
| code | string | ソースコード本体 |
| scope | string | スコープ（global/class/local） |
| loc | int | 実効行数 |

#### オプショナルフィールド

| Field | Type | 説明 |
|-------|------|------|
| arguments | string[] | 引数リスト |
| return_type | string | 戻り値の型 |
| docstring | string | ドキュメント文字列 |
| comments | string[] | コメント行 |
| modifiers | string[] | 修飾子 |
| imports | string[] | インポート/include |
| calls | string[] | 呼び出す関数 |
| complexity | int | サイクロマティック複雑度 |
| comment_lines | int | コメント行数 |

#### 実装マッピング

| 実装 | ファイル | 型 |
|------|---------|-----|
| Go | pkg/models/function_info.go | struct |
| Python (MCP) | mcp-server/server/models.py | dataclass |
| Qdrant | - | JSON payload |

---

## 4. Protocol Buffers定義

### 4.1 サービス定義

**ファイル**: `proto/embedding.proto`

#### EmbeddingService

| RPC | リクエスト | レスポンス | 説明 |
|-----|----------|----------|------|
| EmbedBatch | EmbedBatchRequest | EmbedBatchResponse | バッチベクトル化 |
| Health | HealthRequest | HealthResponse | ヘルスチェック |
| ListModels | ListModelsRequest | ListModelsResponse | モデル一覧取得 |

### 4.2 メッセージ定義

#### EmbedBatchRequest

| Field | Type | 説明 |
|-------|------|------|
| texts | string[] | ベクトル化するテキスト |
| model_name | string | モデル名（オプション） |
| max_length | int32 | 最大トークン長（オプション） |

#### EmbedBatchResponse

| Field | Type | 説明 |
|-------|------|------|
| vectors | Vector[] | ベクトルのリスト |
| model_used | string | 使用されたモデル名 |

#### Vector

| Field | Type | 説明 |
|-------|------|------|
| values | float[] | ベクトル値（768次元） |
| tokens_used | int32 | 使用トークン数 |

#### HealthResponse

| Field | Type | 説明 |
|-------|------|------|
| healthy | bool | サーバー状態 |
| loaded_models | string[] | ロード済みモデル |
| device | string | 使用デバイス（cuda/cpu） |

#### ModelInfo

| Field | Type | 説明 |
|-------|------|------|
| name | string | モデル名 |
| dimension | int32 | ベクトル次元数 |
| max_length | int32 | 最大トークン長 |
| loaded | bool | キャッシュ済みか |
| available | bool | 利用可能か |

### 4.3 コード生成

| ターゲット | 出力先 | ツール |
|----------|--------|--------|
| Go | indexer/proto/embedding/ | protoc-gen-go, protoc-gen-go-grpc |
| Python (Embedding) | embedding-server/proto/embedding/ | grpc_tools.protoc |
| Python (MCP) | mcp-server/proto/embedding/ | grpc_tools.protoc |

**生成コマンド**: `make proto`

---

## 5. Indexer設計（Go）

### 5.1 アーキテクチャ

```
main.go (Orchestrator)
    ↓
    ├→ Config Loader (YAML読み込み)
    ├→ Logger (ロギング設定)
    ├→ File Scanner (ファイル検出)
    ├→ Parser Factory
    │    ├→ Python Parser
    │    ├→ Rust Parser
    │    ├→ Go Parser
    │    ├→ Java Parser
    │    ├→ C Parser
    │    └→ C++ Parser
    ├→ Embedder Client (gRPC)
    └→ Qdrant Indexer (HTTP)
```

### 5.2 モジュール構成

#### cmd/indexer/main.go

**責務**: プログラムのエントリーポイント、全体フロー制御

**主要機能**:
- コマンドライン引数解析（cobra）
- 各モジュールの初期化
- フェーズ管理（スキャン→解析→登録）
- エラーハンドリング
- 進捗表示

**依存**: config, logger, scanner, parser, embedder, indexer

#### internal/config/

**責務**: 設定管理

**モジュール**:
- `config.go`: 設定構造体定義
- `loader.go`: YAML読み込み、環境変数展開、バリデーション

**データ構造**:
- Config（全体設定）
- InputConfig（入力設定）
- QdrantConfig（Qdrant設定）
- EmbeddingServerConfig（埋め込みサーバー設定）
- ProcessingConfig（処理設定）
- LoggingConfig（ログ設定）

#### internal/scanner/

**責務**: ファイル検出とフィルタリング

**モジュール**:
- `scanner.go`: ディレクトリ走査、言語判定
- `ignore.go`: .ragignore処理

**主要機能**:
- 再帰的ディレクトリ走査
- .ragignoreパターンマッチング
- バイナリファイル検出
- 言語別拡張子フィルタリング

**依存**: go-gitignore

#### internal/parser/

**責務**: AST解析によるコード構造抽出

**モジュール**:
- `parser.go`: Parserインターフェース定義
- `function_info.go`: FunctionInfo構造体
- `python_parser.go`: Pythonパーサー
- `rust_parser.go`: Rustパーサー
- `go_parser.go`: Goパーサー
- `java_parser.go`: Javaパーサー
- `c_parser.go`: Cパーサー
- `cpp_parser.go`: C++パーサー
- `factory.go`: パーサーファクトリ

**Parserインターフェース**:
```go
type Parser interface {
    ParseFile(filePath string) ([]*FunctionInfo, error)
    Language() string
}
```

**依存**: go-tree-sitter

#### internal/embedder/

**責務**: 埋め込みサーバーとのgRPC通信

**モジュール**:
- `client.go`: gRPCクライアント実装
- `models.go`: モデル情報管理

**主要機能**:
- gRPC接続管理
- ヘルスチェック
- バッチベクトル化リクエスト
- リトライ処理
- タイムアウト管理

**依存**: grpc, proto/embedding

#### internal/indexer/

**責務**: Qdrantへのデータ登録

**モジュール**:
- `qdrant.go`: Qdrantクライアント、登録処理

**主要機能**:
- コレクション作成・削除
- ベクトルとペイロードの登録
- バッチアップサート
- エラーリトライ

**依存**: qdrant-client

#### internal/logger/

**責務**: ロギング

**モジュール**:
- `logger.go`: ロガー初期化、設定

**主要機能**:
- ファイル・コンソール両方への出力
- ログレベル管理
- フォーマット統一

**依存**: logrus

### 5.3 並列処理戦略

| 処理 | 並列化 | 方式 | 並列度 |
|------|--------|------|--------|
| ファイルスキャン | なし | 順次 | - |
| AST解析 | あり | Goroutine | 設定可能（デフォルト: CPU数） |
| ベクトル化 | バッチ | gRPC Batch | バッチサイズ固定 |
| Qdrant登録 | バッチ | HTTP Batch | バッチサイズ固定 |

---

## 6. Embedding Server設計（Python）

### 6.1 アーキテクチャ

```
main.py (gRPC Server)
    ↓
    ├→ Config Loader
    ├→ Servicer (gRPC Handler)
    │    └→ Model Manager
    │         ├→ Model Cache (Dict)
    │         └→ Jina Embeddings v2
    └→ gRPC Server
```

### 6.2 モジュール構成

#### server/main.py

**責務**: gRPCサーバーのエントリーポイント

**主要機能**:
- gRPCサーバー起動
- Servicer登録
- ポート・ホスト設定
- グレースフルシャットダウン

#### server/servicer.py

**責務**: gRPC Servicerの実装

**実装RPC**:
- EmbedBatch: バッチベクトル化処理
- Health: ヘルスチェック応答
- ListModels: モデル一覧返却

**エラーハンドリング**:
- gRPC StatusCode設定
- ログ出力

#### server/model_manager.py

**責務**: モデルのライフサイクル管理

**主要機能**:
- モデルの遅延ロード（初回アクセス時）
- モデルキャッシュ（メモリ保持）
- GPU/CPU自動切り替え
- 複数モデル対応

**データ構造**:
- EmbeddingModel: モデルラッパークラス
- ModelManager: モデル管理クラス

**依存**: transformers, torch

#### server/config.py

**責務**: 設定管理

**データ構造**:
- ServerConfig: サーバー設定
- ModelConfig: モデル設定
- Config: 全体設定

**主要機能**:
- YAML読み込み
- デフォルト値設定
- バリデーション

---

## 7. MCP Server設計（Python）

### 7.1 アーキテクチャ

```
main.py (MCP Server)
    ↓
    ├→ Config Loader
    ├→ MCP Tools
    │    ├→ search_code
    │    ├→ get_function_details
    │    └→ list_collections
    ├→ Services
    │    ├→ Search Service
    │    │    ├→ Embedder Client (gRPC)
    │    │    └→ Qdrant Client (HTTP)
    │    └→ Collection Service
    │         └→ Qdrant Client (HTTP)
    └→ MCP Protocol Handler
```

### 7.2 モジュール構成

#### server/main.py

**責務**: MCPサーバーのエントリーポイント

**主要機能**:
- MCP Serverインスタンス化
- ツール登録
- stdio通信設定
- リクエストルーティング

**MCPハンドラ**:
- `list_tools()`: 利用可能ツール一覧
- `call_tool()`: ツール実行

#### server/config.py

**責務**: 設定管理

**データ構造**:
- QdrantConfig: Qdrant接続設定
- EmbeddingServerConfig: 埋め込みサーバー設定
- SearchConfig: 検索設定
- LoggingConfig: ログ設定
- Config: 全体設定

#### server/models.py

**責務**: データモデル定義

**FunctionInfoクラス**:
- `from_qdrant_payload()`: Qdrantペイロードから復元
- `to_mcp_response()`: MCP Response形式に変換
- `to_context_text()`: LLMコンテキスト用テキスト生成

**注**: Go側のFunctionInfo構造体と同じフィールド構成

#### server/tools/

**責務**: MCPツールの実装

**search_code.py**:
- 入力: query, collection, limit, filters
- 出力: 検索結果（JSON）
- 処理: クエリベクトル化 → Qdrant検索 → 結果変換

**get_function_details.py**:
- 入力: collection, file_path, start_line
- 出力: 関数詳細（JSON）
- 処理: Qdrant完全一致検索 → 詳細情報返却

**list_collections.py**:
- 入力: なし
- 出力: コレクション一覧（JSON）
- 処理: Qdrant API呼び出し → 統計情報取得

#### server/services/search_service.py

**責務**: 検索ビジネスロジック

**主要機能**:
- クエリのベクトル化
- フィルタ構築
- Qdrant検索実行
- 結果のランキング
- FunctionInfo復元

**依存**: embedder_client, qdrant_client

#### server/services/collection_service.py

**責務**: コレクション管理

**主要機能**:
- コレクション一覧取得
- コレクション統計取得

#### server/clients/embedder_client.py

**責務**: 埋め込みサーバーのgRPCクライアント

**主要機能**:
- gRPC接続管理
- ヘルスチェック
- 単一/バッチベクトル化
- リトライ処理

**依存**: grpc, proto/embedding

#### server/clients/qdrant_client.py

**責務**: QdrantのHTTPクライアント

**主要機能**:
- ベクトル検索
- スクロール検索（フィルタのみ）
- コレクション一覧取得
- ヘルスチェック

**依存**: qdrant-client

### 7.3 MCP Tools仕様

#### search_code

**入力スキーマ**:
```json
{
  "query": "string (required)",
  "collection": "string (required)",
  "limit": "integer (optional, default: 10)",
  "language": "string (optional)",
  "min_complexity": "integer (optional)",
  "max_complexity": "integer (optional)"
}
```

**出力スキーマ**:
```json
{
  "results": [FunctionInfo],
  "total": "integer",
  "query": "string",
  "collection": "string"
}
```

#### get_function_details

**入力スキーマ**:
```json
{
  "collection": "string (required)",
  "file_path": "string (required)",
  "start_line": "integer (required)"
}
```

**出力スキーマ**: FunctionInfo（全フィールド）

#### list_collections

**入力スキーマ**: なし

**出力スキーマ**:
```json
{
  "collections": [
    {
      "name": "string",
      "vector_count": "integer",
      "vector_size": "integer"
    }
  ],
  "total": "integer"
}
```

---

## 8. 依存関係管理

### 8.1 Go依存パッケージ

**ファイル**: `indexer/go.mod`

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| go-tree-sitter | latest | AST解析 |
| qdrant/go-client | v1.7+ | Qdrant操作 |
| grpc | v1.60+ | gRPC通信 |
| protobuf | v1.31+ | Protocol Buffers |
| yaml.v3 | v3.0+ | YAML読み込み |
| logrus | v1.9+ | ロギング |
| cobra | v1.8+ | CLI |
| go-gitignore | latest | .ragignore処理 |

### 8.2 Python依存パッケージ（Embedding Server）

**ファイル**: `embedding-server/requirements.txt`

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| grpcio | ≥1.60.0 | gRPCサーバー |
| grpcio-tools | ≥1.60.0 | protoコンパイル |
| transformers | ≥4.35.0 | 埋め込みモデル |
| torch | ≥2.2.0 | PyTorch |
| PyYAML | ≥6.0 | YAML読み込み |
| pytest | ≥7.4.0 | テスト |
| pytest-asyncio | ≥0.21.0 | 非同期テスト |

### 8.3 Python依存パッケージ（MCP Server）

**ファイル**: `mcp-server/requirements.txt`

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| mcp | ≥0.1.0 | MCP SDK |
| grpcio | ≥1.60.0 | gRPCクライアント |
| grpcio-tools | ≥1.60.0 | protoコンパイル |
| qdrant-client | ≥1.7.0 | Qdrantクライアント |
| PyYAML | ≥6.0 | YAML読み込み |
| pytest | ≥7.4.0 | テスト |
| pytest-asyncio | ≥0.21.0 | 非同期テスト |

---

## 9. ビルド・デプロイ

### 9.1 ビルドターゲット

**Makefile**（ルート）:

| ターゲット | 説明 | 対象 |
|----------|------|------|
| proto | Protocol Buffersコード生成 | 全コンポーネント |
| build | Indexerビルド（ローカル） | Go |
| build-all | 全プラットフォーム向けビルド | Go |
| test | 全コンポーネントテスト | Go + Python |
| docker-build | Dockerイメージビルド | Embedding + MCP |
| docker-up | Docker環境起動 | 全サービス |
| clean | 生成ファイル削除 | 全コンポーネント |

### 9.2 クロスコンパイル

**対象**: Indexer（Go）

| プラットフォーム | バイナリ名 |
|---------------|----------|
| Linux (amd64) | code-rag-indexer-linux-amd64 |
| Linux (arm64) | code-rag-indexer-linux-arm64 |
| Windows (amd64) | code-rag-indexer-windows-amd64.exe |
| macOS (amd64) | code-rag-indexer-darwin-amd64 |
| macOS (arm64) | code-rag-indexer-darwin-arm64 |

**ビルドスクリプト**: `scripts/build-indexer.sh`

### 9.3 Docker構成

**docker-compose.yml**:

| サービス | イメージ | ポート | 依存 |
|---------|---------|--------|------|
| qdrant | qdrant/qdrant:latest | 6333, 6334 | - |
| embedding-server | embedding-server:latest | 50051 | qdrant |
| mcp-server | mcp-server:latest | - (stdio) | qdrant, embedding-server |

**Dockerfile**:
- `embedding-server/Dockerfile`: Python 3.11ベース
- `mcp-server/Dockerfile`: Python 3.11ベース

### 9.4 デプロイメント方式

| コンポーネント | 方式 | 環境 |
|--------------|------|------|
| Indexer | シングルバイナリ | ローカル実行 |
| Embedding Server | Docker Compose | Docker環境 |
| MCP Server | Docker Compose | Docker環境 |
| Qdrant | Docker Compose | Docker環境 |

---

## 10. 設定ファイル

### 10.1 Indexer設定

**ファイル**: `config.yaml`

**構造**:
```yaml
input:
  source_dir: string
  ignore_file: string

qdrant:
  url: string
  api_key: string (env var supported)
  collection_name: string

embedding_server:
  address: string
  timeout: duration
  max_retries: int
  batch_size: int
  model_name: string (optional)

processing:
  parallel_workers: int
  languages: string[]

logging:
  level: string
  file: string
```

### 10.2 Embedding Server設定

**ファイル**: `embedding-server.yaml`

**構造**:
```yaml
server:
  host: string
  port: int
  max_workers: int

default_model: string

models:
  - name: string
    dimension: int
    max_length: int
    trust_remote_code: bool
    preload: bool

device: string (auto/cuda/cpu)

logging_level: string
logging_file: string (optional)
```

### 10.3 MCP Server設定

**ファイル**: `mcp-server.yaml`

**構造**:
```yaml
qdrant:
  url: string
  api_key: string (env var supported)

embedding_server:
  address: string
  timeout: int
  max_retries: int
  model_name: string

search:
  default_limit: int
  max_limit: int
  timeout: int

logging:
  level: string
  file: string (optional)
```

### 10.4 Claude Desktop設定

**ファイル**: `claude_desktop_config.json`

**Docker実行**:
```json
{
  "mcpServers": {
    "code-rag": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--network", "host",
        "-v", "/path/to/mcp-server.yaml:/app/config.yaml:ro",
        "mcp-server:latest"
      ]
    }
  }
}
```

**ローカル実行**:
```json
{
  "mcpServers": {
    "code-rag": {
      "command": "python",
      "args": ["-m", "server.main"],
      "cwd": "/path/to/code-rag-indexer/mcp-server",
      "env": {
        "MCP_SERVER_CONFIG": "/path/to/mcp-server.yaml",
        "PYTHONPATH": "/path/to/code-rag-indexer/mcp-server"
      }
    }
  }
}
```

---

## 11. インターフェース仕様

### 11.1 CLI（Indexer）

**コマンド**: `code-rag-indexer`

**オプション**:
| オプション | 短縮 | 必須 | 説明 |
|----------|------|------|------|
| --config | -c | Yes | 設定ファイルパス |
| --verbose | -v | No | 詳細ログ出力 |
| --version | - | No | バージョン表示 |
| --help | -h | No | ヘルプ表示 |

**終了コード**:
| コード | 意味 |
|--------|------|
| 0 | 正常終了 |
| 1 | エラー終了 |

### 11.2 gRPC API（Embedding Server）

**サービス**: EmbeddingService

**エンドポイント**: `localhost:50051`

**RPC一覧**:
- `EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse)`
- `Health(HealthRequest) returns (HealthResponse)`
- `ListModels(ListModelsRequest) returns (ListModelsResponse)`

### 11.3 MCP API（MCP Server）

**プロトコル**: MCP over stdio

**ツール一覧**:
- `search_code`: コード検索
- `get_function_details`: 関数詳細取得
- `list_collections`: コレクション一覧

---

## 12. エラーハンドリング戦略

### 12.1 Indexer

| エラー種別 | 対応 |
|----------|------|
| 設定ファイル読み込み失敗 | 即座に終了、エラーメッセージ表示 |
| 埋め込みサーバー接続失敗 | リトライ後、終了 |
| ファイル解析失敗 | 警告ログ、スキップして継続 |
| Qdrant登録失敗 | リトライ後、終了 |

### 12.2 Embedding Server

| エラー種別 | 対応 |
|----------|------|
| モデルロード失敗 | gRPC INTERNAL、詳細ログ |
| 不正なリクエスト | gRPC INVALID_ARGUMENT |
| リソース不足 | gRPC RESOURCE_EXHAUSTED |

### 12.3 MCP Server

| エラー種別 | 対応 |
|----------|------|
| 埋め込みサーバー接続失敗 | エラーレスポンス、リトライ提案 |
| Qdrant接続失敗 | エラーレスポンス |
| 検索結果なし | 空の結果配列、エラーなし |
| タイムアウト | エラーレスポンス、タイムアウト明記 |

---

## 13. ログ出力仕様

### 13.1 ログレベル

| レベル | 用途 |
|--------|------|
| DEBUG | 詳細なデバッグ情報 |
| INFO | 通常の処理情報 |
| WARN | 警告（処理継続可能） |
| ERROR | エラー（処理中断） |

### 13.2 ログフォーマット

**形式**: `[TIMESTAMP] [LEVEL] [COMPONENT] MESSAGE`

**例**:
```
[2025-11-24 10:30:45] [INFO] [Scanner] Found 150 files
[2025-11-24 10:30:46] [WARN] [Parser] Skipped file: syntax error in main.py
[2025-11-24 10:30:50] [INFO] [Indexer] Registered 1000 functions
[2025-11-24 10:30:51] [ERROR] [EmbedderClient] Connection failed: timeout
```

### 13.3 ログ出力先

| コンポーネント | 出力先 |
|--------------|--------|
| Indexer | コンソール + ファイル（設定可能） |
| Embedding Server | コンソール + ファイル（設定可能） |
| MCP Server | ファイル（設定可能） |

---

## 14. パフォーマンス目標

### 14.1 Indexer

| 指標 | 目標値 |
|------|--------|
| 100万行のコードベース処理時間 | 8時間以内 |
| メモリ使用量 | 32GB以下 |
| 並列度 | CPU数に応じて自動調整 |

### 14.2 Embedding Server

| 指標 | 目標値 |
|------|--------|
| バッチベクトル化（8件） | 1秒以内 |
| メモリ使用量（モデル1つ） | 2-4GB |
| 同時リクエスト処理 | 10+ |

### 14.3 MCP Server

| 指標 | 目標値 |
|------|--------|
| 検索レスポンスタイム（P95） | 1秒以内 |
| メモリ使用量 | 200MB以下 |
| 同時検索処理 | 10 req/sec |

---

## 15. セキュリティ考慮事項

### 15.1 認証・認可

| コンポーネント | 認証方式 |
|--------------|---------|
| Qdrant | APIキー認証（オプション） |
| Embedding Server (gRPC) | なし（ローカル通信想定） |
| MCP Server | なし（Claude Desktop経由のみ） |

### 15.2 データ保護

| 項目 | 対応 |
|------|------|
| APIキー | 環境変数から読み込み、平文保存回避 |
| ソースコード | Qdrant内に保存、アクセス制御はQdrantに依存 |
| gRPC通信 | TLS未使用（ローカル通信のみ） |

---

## 16. テスト戦略

### 16.1 単体テスト

| コンポーネント | フレームワーク | カバレッジ目標 |
|--------------|--------------|--------------|
| Indexer (Go) | go test | >80% |
| Embedding Server (Python) | pytest | >85% |
| MCP Server (Python) | pytest | >85% |

### 16.2 統合テスト

| テストケース | 対象 |
|------------|------|
| Indexer → Embedding Server | gRPC通信 |
| Indexer → Qdrant | データ登録 |
| MCP Server → Embedding Server | gRPC通信 |
| MCP Server → Qdrant | 検索 |

### 16.3 E2Eテスト

| シナリオ | 説明 |
|---------|------|
| 完全フロー | Indexer実行 → MCP検索 → 結果確認 |
| エラーケース | 接続失敗、データなし等 |

---

この基本設計書に基づいて、TDD方式で各モジュールを実装してください。実装の詳細なロジックは各実装フェーズで決定します。