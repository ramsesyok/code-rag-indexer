# Code RAG System 基本設計書

**文書バージョン**: 1.0  
**作成日**: 2025-11-25  
**プロジェクト名**: Code RAG System (Indexer + Embedding Server + MCP Server)  
**対応要件定義書**: requirements.md v3.0

---

## 目次

1. [はじめに](#1-はじめに)
2. [システム概要](#2-システム概要)
3. [システム構成](#3-システム構成)
4. [機能設計](#4-機能設計)
5. [データ設計](#5-データ設計)
6. [インターフェース設計](#6-インターフェース設計)
7. [エラー処理設計](#7-エラー処理設計)
8. [性能設計](#8-性能設計)
9. [セキュリティ設計](#9-セキュリティ設計)
10. [運用設計](#10-運用設計)
11. [制約事項](#11-制約事項)

---

## 1. はじめに

### 1.1 文書の目的

本文書は、Code RAG Systemの基本設計を定義し、開発チーム間での共通理解を形成することを目的とする。

### 1.2 対象読者

- システム開発者
- テスト担当者
- プロジェクトマネージャー
- システム管理者

### 1.3 前提条件

- 本設計書は要件定義書（requirements.md v3.0）に基づいて作成されている
- IPAの機能要件の合意形成ガイドに準拠している
- 各機能要件には対応する要件IDが記載されている

---

## 2. システム概要

### 2.1 システムの目的

大規模なソースコードベース（10万～100万行超）の理解、リファクタリング支援、バグ調査を効率化するため、ローカルLLMとローカルRAGを活用したコード理解支援システムを構築する。

### 2.2 システム構成要素

| コンポーネント | 実装言語 | 主要機能 |
|--------------|---------|---------|
| **Indexer** | Go | ソースコードのAST解析とベクトル登録 |
| **Embedding Server** | Python | コード埋め込みベクトル生成（gRPC） |
| **MCP Server** | Python | LLMからの検索インターフェース（MCP） |
| **Qdrant** | - | ベクトルデータベース（外部製品） |

### 2.3 システム動作フロー

#### 2.3.1 登録フロー

```
[ソースコード] → [Indexer] → [Embedding Server] → [Qdrant]
                      ↓
               [AST解析/メタデータ抽出]
```

#### 2.3.2 検索フロー

```
[LLM/Claude] ↔ [MCP Server] ↔ [Embedding Server]
                     ↓
                 [Qdrant]
```

---

## 3. システム構成

### 3.1 物理構成

```
┌─────────────────────────────────────────────┐
│         開発者のローカル環境                  │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  Claude Desktop / LLM Client         │   │
│  └──────────┬───────────────────────────┘   │
│             │ stdio (MCP)                   │
│  ┌──────────▼───────────────────────────┐   │
│  │  MCP Server (Docker)                 │   │
│  │  - Port: stdio                       │   │
│  └──────┬───────────────────┬───────────┘   │
│         │ gRPC              │ HTTP          │
│  ┌──────▼──────────┐  ┌────▼──────────┐    │
│  │ Embedding Server│  │ Qdrant        │    │
│  │ (Docker)        │  │ (Docker)      │    │
│  │ Port: 50051     │  │ Port: 6333    │    │
│  └─────────────────┘  └───────────────┘    │
│         ▲                     ▲             │
│         │ gRPC                │ HTTP        │
│  ┌──────┴─────────────────────┴─────────┐  │
│  │  Indexer (Native Binary)             │  │
│  │  - Windows/Linux/macOS               │  │
│  └──────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

### 3.2 ネットワーク構成

| 通信経路 | プロトコル | ポート | 用途 |
|---------|----------|--------|------|
| Claude Desktop ↔ MCP Server | stdio | - | MCP Protocol |
| MCP Server → Qdrant | HTTP | 6333 | ベクトル検索 |
| MCP Server → Embedding Server | gRPC | 50051 | クエリベクトル化 |
| Indexer → Qdrant | HTTP | 6333 | ベクトル登録 |
| Indexer → Embedding Server | gRPC | 50051 | コードベクトル化 |

---

## 4. 機能設計

### 4.1 Indexer（Go実装）

#### 4.1.1 機能一覧

| 機能ID | 機能名 | 対応要件ID | 概要 |
|--------|--------|-----------|------|
| IDX-F01 | ファイル入力機能 | IDX-IN-001, IDX-IN-003 | ソースコードディレクトリを入力として受け付ける |
| IDX-F02 | 除外パターン処理 | IDX-IN-101, IDX-IN-102, IDX-IN-103 | .ragignoreやバイナリファイルを除外 |
| IDX-F03 | AST解析機能 | IDX-AST-001～004 | Tree-sitterによる構文解析 |
| IDX-F04 | メタデータ抽出機能 | IDX-AST-101～113 | 関数情報・依存関係の抽出 |
| IDX-F05 | ベクトル化要求機能 | IDX-VEC-001～008 | Embedding ServerへのgRPC通信 |
| IDX-F06 | Qdrant登録機能 | IDX-QDR-001～003 | ベクトルとメタデータの登録 |
| IDX-F07 | コレクション管理機能 | IDX-COL-001～005 | コレクションの作成・削除 |
| IDX-F08 | 並列処理機能 | IDX-PRF-001 | Goroutineによる並列実行 |

#### 4.1.2 IDX-F01: ファイル入力機能

**対応要件ID**: IDX-IN-001, IDX-IN-003

**機能概要**:
- ローカルディレクトリを再帰的にスキャン
- YAML設定ファイルから入力パスを読み込み
- 対象言語の拡張子を持つファイルを選別

**入力**:
```go
type InputConfig struct {
    SourceDir  string   `yaml:"source_dir"`
    IgnoreFile string   `yaml:"ignore_file"`
}
```

**出力**:
```go
type FileInfo struct {
    Path         string
    Language     string
    RelativePath string
    Size         int64
}
```

**処理フロー**:
1. 設定ファイルからSourceDirを読み込み
2. filepath.Walkによる再帰的スキャン
3. 拡張子による言語判定
4. FileInfo配列を生成

#### 4.1.3 IDX-F02: 除外パターン処理

**対応要件ID**: IDX-IN-101, IDX-IN-102, IDX-IN-103

**機能概要**:
- .ragignoreファイルのパース（.gitignore互換）
- バイナリファイルの自動検出
- 設定による拡張子フィルタリング

**データ構造**:
```go
type ExclusionRules struct {
    Patterns      []string // .ragignoreから読み込み
    BinaryCheck   bool
    ExcludeExts   []string // 設定ファイルから
}
```

**処理フロー**:
1. .ragignoreファイルの読み込み
2. gitignoreライブラリによるパターンマッチング
3. バイナリ判定（先頭1KBのNULLバイト検出）
4. 除外判定の実行

#### 4.1.4 IDX-F03: AST解析機能

**対応要件ID**: IDX-AST-001～004

**機能概要**:
- Tree-sitterによる多言語対応パース
- 関数・メソッド単位での分割
- 構文エラーの検出とスキップ

**データ構造**:
```go
type ASTParser interface {
    Parse(source []byte) (*sitter.Tree, error)
    ExtractFunctions(tree *sitter.Tree) ([]FunctionNode, error)
}

type FunctionNode struct {
    Node        *sitter.Node
    StartByte   uint32
    EndByte     uint32
    StartPoint  sitter.Point
    EndPoint    sitter.Point
}
```

**対応言語とパーサー**:
| 言語 | Tree-sitter Grammar |
|------|-------------------|
| Python | tree-sitter-python |
| Go | tree-sitter-go |
| Rust | tree-sitter-rust |
| Java | tree-sitter-java |
| C | tree-sitter-c |
| C++ | tree-sitter-cpp |

#### 4.1.5 IDX-F04: メタデータ抽出機能

**対応要件ID**: IDX-AST-101～113

**機能概要**:
- 関数シグネチャの抽出
- 依存関係の解析
- コメント・ドキュメントの抽出
- 複雑度の計算

**データ構造**:
```go
type FunctionInfo struct {
    // 基本情報 (IDX-AST-101)
    Name          string   `json:"name"`
    Signature     string   `json:"signature"`
    Parameters    []Param  `json:"parameters"`
    ReturnType    string   `json:"return_type"`
    
    // 位置情報 (IDX-AST-107)
    FilePath      string   `json:"file_path"`
    StartLine     int      `json:"start_line"`
    EndLine       int      `json:"end_line"`
    StartColumn   int      `json:"start_column"`
    EndColumn     int      `json:"end_column"`
    
    // クラス情報 (IDX-AST-102)
    ClassName     string   `json:"class_name,omitempty"`
    ClassHierarchy []string `json:"class_hierarchy,omitempty"`
    
    // 依存関係 (IDX-AST-103, 108, 109)
    Imports       []string `json:"imports"`
    CalledFuncs   []string `json:"called_functions"`
    CalledBy      []string `json:"called_by,omitempty"`
    
    // ドキュメント (IDX-AST-104)
    Docstring     string   `json:"docstring,omitempty"`
    Comments      []string `json:"comments,omitempty"`
    
    // スコープ・修飾子 (IDX-AST-105, 106)
    Scope         string   `json:"scope"` // global/class/local
    Modifiers     []string `json:"modifiers"` // public/private/static等
    
    // 使用型 (IDX-AST-110)
    UsedTypes     []string `json:"used_types"`
    
    // メトリクス (IDX-AST-111, 112)
    Complexity    int      `json:"complexity,omitempty"`
    LOC           int      `json:"loc"`
    CommentLines  int      `json:"comment_lines"`
    
    // 言語固有 (IDX-AST-113)
    Decorators    []string `json:"decorators,omitempty"`    // Python
    Annotations   []string `json:"annotations,omitempty"`   // Java
    Attributes    []string `json:"attributes,omitempty"`    // C#
    
    // コード本体
    Body          string   `json:"body"`
    Language      string   `json:"language"`
}

type Param struct {
    Name string `json:"name"`
    Type string `json:"type,omitempty"`
}
```

**処理フロー**:
1. Tree-sitterのNodeからシグネチャ抽出
2. パラメータと戻り値のパース
3. クラス階層の走査
4. import/includeステートメントの検出
5. 関数呼び出しの解析
6. コメントノードの収集
7. サイクロマティック複雑度の計算
8. 行数のカウント

#### 4.1.6 IDX-F05: ベクトル化要求機能

**対応要件ID**: IDX-VEC-001～008

**機能概要**:
- Embedding ServerへのgRPC接続
- バッチリクエストの送信
- リトライ処理
- ヘルスチェック

**gRPCインターフェース定義** (Protocol Buffers):
```protobuf
service EmbeddingService {
  rpc EmbedBatch(EmbedRequest) returns (EmbedResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}

message EmbedRequest {
  repeated string texts = 1;
  string model_name = 2;
}

message EmbedResponse {
  repeated Embedding embeddings = 1;
  string model_name = 2;
  int32 dimension = 3;
}

message Embedding {
  repeated float values = 1;
}

message HealthRequest {}

message HealthResponse {
  bool healthy = 1;
  string message = 2;
}

message ListModelsRequest {}

message ListModelsResponse {
  repeated ModelInfo models = 1;
}

message ModelInfo {
  string name = 1;
  int32 dimension = 2;
  int32 max_length = 3;
}
```

**Goクライアント構造**:
```go
type EmbeddingClient struct {
    conn       *grpc.ClientConn
    client     pb.EmbeddingServiceClient
    timeout    time.Duration
    maxRetries int
    batchSize  int
}

func (c *EmbeddingClient) EmbedBatch(
    ctx context.Context,
    texts []string,
    modelName string,
) ([][]float32, error)
```

**処理フロー**:
1. gRPC接続の確立（ダイヤル）
2. ヘルスチェックの実行
3. テキストのバッチ分割（設定サイズ）
4. EmbedBatch RPCの呼び出し
5. エラー発生時のリトライ（最大3回）
6. ベクトルの集約と返却

#### 4.1.7 IDX-F06: Qdrant登録機能

**対応要件ID**: IDX-QDR-001～003

**機能概要**:
- QdrantへのHTTP接続
- ベクトルポイントの一括登録
- ペイロードの構造化

**データ構造**:
```go
type QdrantPoint struct {
    ID      string                 `json:"id"`
    Vector  []float32              `json:"vector"`
    Payload map[string]interface{} `json:"payload"`
}

type QdrantBatchRequest struct {
    Points []QdrantPoint `json:"points"`
}
```

**処理フロー**:
1. FunctionInfoをペイロードに変換
2. UUIDの生成（ファイルパス+関数名ベース）
3. QdrantPointの構築
4. `/collections/{collection}/points` エンドポイントへのPOST
5. レスポンスの検証

#### 4.1.8 IDX-F07: コレクション管理機能

**対応要件ID**: IDX-COL-001～005

**機能概要**:
- コレクションの作成
- 既存コレクションの削除
- ベクトル次元の設定

**APIエンドポイント**:
```
PUT /collections/{collection_name}
DELETE /collections/{collection_name}
GET /collections/{collection_name}
```

**コレクション設定**:
```go
type CollectionConfig struct {
    VectorSize uint64 `json:"vector_size"` // 768
    Distance   string `json:"distance"`    // Cosine
}
```

**処理フロー**:
1. コレクション存在確認（GET）
2. 既存の場合は削除（DELETE）
3. 新規作成（PUT）
4. 設定の検証

#### 4.1.9 IDX-F08: 並列処理機能

**対応要件ID**: IDX-PRF-001

**機能概要**:
- ファイル単位でのGoroutine並列実行
- ワーカープールによる同時実行数制御
- エラー集約

**データ構造**:
```go
type WorkerPool struct {
    numWorkers int
    jobs       chan FileInfo
    results    chan ProcessResult
    errors     chan error
    wg         sync.WaitGroup
}

type ProcessResult struct {
    FilePath  string
    Functions []FunctionInfo
    Vectors   [][]float32
}
```

**処理フロー**:
1. ワーカープールの初期化
2. ファイルリストをjobsチャネルに送信
3. 各ワーカーがjobsを取得して処理
4. 結果をresultsチャネルに送信
5. メインゴルーチンが結果を集約

---

### 4.2 Embedding Server（Python実装）

#### 4.2.1 機能一覧

| 機能ID | 機能名 | 対応要件ID | 概要 |
|--------|--------|-----------|------|
| EMB-F01 | gRPCサーバー起動 | EMB-001 | gRPCサーバーとしての起動 |
| EMB-F02 | モデル管理機能 | EMB-101～105 | モデルの遅延ロードとキャッシュ |
| EMB-F03 | バッチ埋め込み機能 | EMB-003, EMB-002 | 複数テキストの一括ベクトル化 |
| EMB-F04 | ヘルスチェック機能 | EMB-004 | サーバー状態の確認 |
| EMB-F05 | モデル一覧機能 | EMB-005 | サポートモデルの情報提供 |
| EMB-F06 | デバイス管理機能 | EMB-006 | GPU/CPU自動切り替え |
| EMB-F07 | エラーハンドリング | EMB-007 | gRPCステータスコード返却 |
| EMB-F08 | ロギング機能 | EMB-008 | リクエスト・エラーログ |

#### 4.2.2 EMB-F01: gRPCサーバー起動

**対応要件ID**: EMB-001

**機能概要**:
- gRPCサーバーのセットアップ
- サービスの登録
- 指定ポートでのリッスン

**データ構造**:
```python
from concurrent import futures
import grpc
from generated import embedding_pb2_grpc

class EmbeddingServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_manager = ModelManager(config)
```

**処理フロー**:
1. 設定ファイルの読み込み
2. gRPCサーバーの生成
3. EmbeddingServicerの登録
4. ポートバインドとサーバー起動

#### 4.2.3 EMB-F02: モデル管理機能

**対応要件ID**: EMB-101～105

**機能概要**:
- モデルの遅延ロード
- ロード済みモデルのキャッシュ
- デフォルトモデルの管理
- プリロード対応

**データ構造**:
```python
from dataclasses import dataclass
from typing import Dict, Optional
from transformers import AutoModel, AutoTokenizer

@dataclass
class ModelConfig:
    name: str
    dimension: int
    max_length: int
    trust_remote_code: bool
    preload: bool

class ModelManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._models: Dict[str, AutoModel] = {}
        self._tokenizers: Dict[str, AutoTokenizer] = {}
        self.default_model = config.default_model
        
    def load_model(self, model_name: str) -> tuple[AutoModel, AutoTokenizer]:
        """遅延ロード: 初回アクセス時のみロード"""
        pass
        
    def get_model(self, model_name: Optional[str] = None) -> tuple[AutoModel, AutoTokenizer]:
        """キャッシュからモデル取得、なければロード"""
        pass
```

**処理フロー**:
1. モデル名の解決（指定なしの場合はデフォルト）
2. キャッシュ確認
3. キャッシュミスの場合はHugging Faceからロード
4. キャッシュに保存
5. モデルとトークナイザーを返却

#### 4.2.4 EMB-F03: バッチ埋め込み機能

**対応要件ID**: EMB-003, EMB-002

**機能概要**:
- 複数テキストの一括ベクトル化
- トークン化とバッチ処理
- mean poolingによるベクトル集約

**実装例**:
```python
def EmbedBatch(self, request, context):
    try:
        model, tokenizer = self.model_manager.get_model(request.model_name)
        
        # トークン化
        inputs = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(self.device)
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # レスポンス構築
        response = embedding_pb2.EmbedResponse()
        for emb in embeddings:
            embedding_msg = response.embeddings.add()
            embedding_msg.values.extend(emb.cpu().numpy().tolist())
        
        return response
        
    except Exception as e:
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(e))
        return embedding_pb2.EmbedResponse()
```

#### 4.2.5 EMB-F04: ヘルスチェック機能

**対応要件ID**: EMB-004

**機能概要**:
- サーバーの稼働状態確認
- モデルロード状態の確認

**実装例**:
```python
def Health(self, request, context):
    try:
        # デフォルトモデルがロード可能か確認
        self.model_manager.get_model()
        
        return embedding_pb2.HealthResponse(
            healthy=True,
            message="Server is healthy"
        )
    except Exception as e:
        return embedding_pb2.HealthResponse(
            healthy=False,
            message=f"Error: {str(e)}"
        )
```

#### 4.2.6 EMB-F05: モデル一覧機能

**対応要件ID**: EMB-005

**機能概要**:
- サポートモデルの情報提供
- 次元数・最大長の返却

**実装例**:
```python
def ListModels(self, request, context):
    response = embedding_pb2.ListModelsResponse()
    
    for model_config in self.config.models:
        model_info = response.models.add()
        model_info.name = model_config.name
        model_info.dimension = model_config.dimension
        model_info.max_length = model_config.max_length
    
    return response
```

#### 4.2.7 EMB-F06: デバイス管理機能

**対応要件ID**: EMB-006

**機能概要**:
- GPU利用可能時はCUDAデバイスを使用
- GPU未検出時はCPUにフォールバック

**実装例**:
```python
import torch

class DeviceManager:
    def __init__(self, device_config: str):
        self.device = self._detect_device(device_config)
    
    def _detect_device(self, config: str) -> torch.device:
        if config == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            return torch.device("cuda")
        else:
            return torch.device("cpu")
```

---

### 4.3 MCP Server（Python実装)

#### 4.3.1 機能一覧

| 機能ID | 機能名 | 対応要件ID | 概要 |
|--------|--------|-----------|------|
| MCP-F01 | MCP Protocol対応 | MCP-001～004 | JSON-RPC over stdioの実装 |
| MCP-F02 | search_codeツール | MCP-SEARCH-001～007 | コード検索機能 |
| MCP-F03 | get_function_detailsツール | MCP-DETAIL-001～003 | 関数詳細取得 |
| MCP-F04 | list_collectionsツール | MCP-LIST-001～003 | コレクション一覧 |
| MCP-F05 | クエリベクトル化 | - | Embedding Server連携 |
| MCP-F06 | Qdrant検索実行 | - | ベクトル検索とフィルタリング |

#### 4.3.2 MCP-F01: MCP Protocol対応

**対応要件ID**: MCP-001～004

**機能概要**:
- stdioベースのJSON-RPC通信
- ツール一覧の提供
- リクエストルーティング

**データ構造**:
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class MCPRequest:
    method: str
    params: Dict[str, Any]
    id: Optional[int] = None

@dataclass
class MCPResponse:
    result: Any
    id: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
```

**処理フロー**:
1. stdinからJSON-RPCリクエストを読み込み
2. メソッドに応じてツール実行
3. 結果をJSON-RPCレスポンスに変換
4. stdoutに出力

#### 4.3.3 MCP-F02: search_codeツール

**対応要件ID**: MCP-SEARCH-001～007

**機能概要**:
- 自然言語クエリによるコード検索
- メタデータフィルタリング
- スコアリングとソート

**入力スキーマ**:
```python
SEARCH_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "自然言語による検索クエリ"
        },
        "collection": {
            "type": "string",
            "description": "検索対象のコレクション名"
        },
        "limit": {
            "type": "integer",
            "description": "取得件数",
            "default": 10,
            "minimum": 1,
            "maximum": 50
        },
        "filters": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "min_complexity": {"type": "integer"},
                "max_complexity": {"type": "integer"},
                "file_pattern": {"type": "string"}
            }
        }
    },
    "required": ["query", "collection"]
}
```

**出力データ構造**:
```python
@dataclass
class SearchResult:
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    signature: str
    body: str
    docstring: Optional[str]
    language: str
    complexity: Optional[int]
```

**処理フロー**:
1. クエリテキストをEmbedding Serverでベクトル化
2. Qdrantフィルタ条件の構築
3. ベクトル検索の実行
4. 結果のスコアリング・ソート
5. 指定件数の結果を返却

**実装例**:
```python
async def search_code(
    query: str,
    collection: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    # クエリベクトル化
    query_vector = await self.embedding_client.embed([query])
    
    # Qdrantフィルタ構築
    qdrant_filter = self._build_filter(filters)
    
    # ベクトル検索
    search_results = await self.qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector[0],
        limit=limit,
        query_filter=qdrant_filter
    )
    
    # 結果変換
    return [self._to_search_result(hit) for hit in search_results]
```

#### 4.3.4 MCP-F03: get_function_detailsツール

**対応要件ID**: MCP-DETAIL-001～003

**機能概要**:
- IDによる関数詳細の取得
- 全メタデータの返却

**入力スキーマ**:
```python
GET_FUNCTION_DETAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "function_id": {
            "type": "string",
            "description": "関数のUUID"
        },
        "collection": {
            "type": "string",
            "description": "コレクション名"
        }
    },
    "required": ["function_id", "collection"]
}
```

**処理フロー**:
1. QdrantからIDでポイント取得
2. ペイロードからFunctionInfo復元
3. JSON形式で返却

#### 4.3.5 MCP-F04: list_collectionsツール

**対応要件ID**: MCP-LIST-001～003

**機能概要**:
- 登録済みコレクションの一覧取得
- 各コレクションの統計情報

**出力データ構造**:
```python
@dataclass
class CollectionInfo:
    name: str
    vectors_count: int
    indexed_at: str
    vector_size: int
```

**処理フロー**:
1. Qdrantからコレクション一覧を取得
2. 各コレクションの統計情報を取得
3. CollectionInfo配列を返却

---

## 5. データ設計

### 5.1 データモデル

#### 5.1.1 FunctionInfoペイロード

**説明**: Qdrantに保存される関数メタデータの完全なスキーマ

**対応要件ID**: IDX-QDR-002, IDX-QDR-003, MCP-003

**データ構造**:
```json
{
  "name": "authenticate_user",
  "signature": "def authenticate_user(username: str, password: str) -> Optional[User]",
  "parameters": [
    {"name": "username", "type": "str"},
    {"name": "password", "type": "str"}
  ],
  "return_type": "Optional[User]",
  
  "file_path": "/src/auth/handlers.py",
  "start_line": 45,
  "end_line": 78,
  "start_column": 0,
  "end_column": 4,
  
  "class_name": "AuthHandler",
  "class_hierarchy": ["BaseHandler", "AuthHandler"],
  
  "imports": ["typing.Optional", "models.User", "bcrypt"],
  "called_functions": ["hash_password", "db.query"],
  "called_by": ["login_endpoint"],
  
  "docstring": "ユーザー認証を実行する...",
  "comments": ["# パスワードのハッシュ化"],
  
  "scope": "class",
  "modifiers": ["public", "async"],
  
  "used_types": ["str", "Optional", "User"],
  
  "complexity": 5,
  "loc": 34,
  "comment_lines": 8,
  
  "decorators": ["@require_auth"],
  
  "body": "def authenticate_user(...):\n    ...",
  "language": "python"
}
```

### 5.2 ベクトル次元

| モデル | 次元数 | 対応要件ID |
|--------|--------|-----------|
| jinaai/jina-embeddings-v2-base-code | 768 | IDX-VEC-002 |

### 5.3 Qdrantコレクション設定

**コレクション作成パラメータ**:
```json
{
  "vector_size": 768,
  "distance": "Cosine",
  "on_disk_payload": false
}
```

**インデックス設定**:
- ペイロードインデックス:
  - `language` (keyword)
  - `complexity` (integer)
  - `file_path` (text)

---

## 6. インターフェース設計

### 6.1 Indexer ↔ Embedding Server

**プロトコル**: gRPC  
**対応要件ID**: IDX-VEC-001

**接続パラメータ**:
```yaml
address: "localhost:50051"
timeout: 30s
max_retries: 3
```

**RPCメソッド**:
| メソッド | 入力 | 出力 | 用途 |
|---------|------|------|------|
| EmbedBatch | texts[], model_name | embeddings[][] | バッチベクトル化 |
| Health | - | healthy, message | ヘルスチェック |
| ListModels | - | models[] | サポートモデル一覧 |

### 6.2 Indexer ↔ Qdrant

**プロトコル**: HTTP REST API  
**対応要件ID**: IDX-QDR-001

**主要エンドポイント**:
| Method | Endpoint | 用途 |
|--------|----------|------|
| PUT | `/collections/{name}` | コレクション作成 |
| DELETE | `/collections/{name}` | コレクション削除 |
| POST | `/collections/{name}/points` | ポイント登録 |
| GET | `/collections/{name}` | コレクション情報取得 |

### 6.3 MCP Server ↔ Embedding Server

**プロトコル**: gRPC  
**対応要件ID**: MCP-F05

**接続パラメータ**:
```yaml
address: "localhost:50051"
timeout: 30s
model_name: "jinaai/jina-embeddings-v2-base-code"
```

**使用RPC**: EmbedBatch

### 6.4 MCP Server ↔ Qdrant

**プロトコル**: HTTP REST API  
**対応要件ID**: MCP-F06

**主要エンドポイント**:
| Method | Endpoint | 用途 |
|--------|----------|------|
| POST | `/collections/{name}/points/search` | ベクトル検索 |
| GET | `/collections/{name}/points/{id}` | ポイント取得 |
| GET | `/collections` | コレクション一覧 |

### 6.5 Claude Desktop ↔ MCP Server

**プロトコル**: MCP (JSON-RPC over stdio)  
**対応要件ID**: MCP-001, MCP-002

**通信方式**:
- 入力: stdin (JSON-RPC request)
- 出力: stdout (JSON-RPC response)

**提供ツール**:
| ツール名 | 説明 |
|---------|------|
| search_code | コード検索 |
| get_function_details | 関数詳細取得 |
| list_collections | コレクション一覧 |

---

## 7. エラー処理設計

### 7.1 Indexerのエラー処理

| エラーカテゴリ | 処理方針 | 対応要件ID |
|--------------|---------|-----------|
| ファイル読み込みエラー | ログ出力してスキップ | - |
| AST解析エラー | 警告出力してスキップ | IDX-AST-003 |
| 埋め込みサーバー接続エラー | リトライ（最大3回）→失敗 | IDX-VEC-006 |
| Qdrant接続エラー | 即座に終了 | - |
| メモリ不足 | エラーメッセージ→終了 | - |

### 7.2 Embedding Serverのエラー処理

| エラーカテゴリ | gRPCステータスコード | 対応要件ID |
|--------------|-------------------|-----------|
| モデルロード失敗 | UNAVAILABLE | EMB-007 |
| トークン数超過 | INVALID_ARGUMENT | EMB-007 |
| GPU OOM | RESOURCE_EXHAUSTED | EMB-007 |
| 不明なモデル名 | NOT_FOUND | EMB-007 |
| 内部エラー | INTERNAL | EMB-007 |

### 7.3 MCP Serverのエラー処理

| エラーカテゴリ | MCPエラーコード | 対応要件ID |
|--------------|---------------|-----------|
| 不正なパラメータ | -32602 (Invalid params) | MCP-004 |
| コレクション未検出 | -32001 (Custom) | MCP-004 |
| 埋め込みサーバーエラー | -32002 (Custom) | MCP-004 |
| Qdrantエラー | -32003 (Custom) | MCP-004 |

---

## 8. 性能設計

### 8.1 処理性能目標

**対応要件ID**: IDX-PRF-001～003

| 項目 | 目標値 |
|------|--------|
| 並列ワーカー数 | 4（設定可能） |
| メモリ使用量 | 32GB以下 |
| 処理時間（100万行） | 8時間以内 |

### 8.2 最適化設計

#### 8.2.1 Indexerの最適化

1. **ファイル単位並列処理**
   - Goroutineによるワーカープール
   - ファイルごとに独立処理

2. **バッチベクトル化**
   - 複数関数をまとめてgRPC送信
   - バッチサイズ: 8（設定可能）

3. **Qdrant一括登録**
   - 100ポイントごとにバッチ登録

#### 8.2.2 Embedding Serverの最適化

1. **モデルキャッシュ**
   - ロード済みモデルをメモリ保持
   - プリロード対応

2. **GPU活用**
   - CUDA利用可能時は自動切り替え
   - バッチ推論の効率化

#### 8.2.3 MCP Serverの最適化

1. **gRPC接続プール**
   - 接続の再利用

2. **Qdrantクライアントキャッシュ**
   - 接続の維持

---

## 9. セキュリティ設計

### 9.1 通信セキュリティ

**対応要件ID**: CNS-002, CNS-003

| 通信経路 | セキュリティ対策 | 備考 |
|---------|---------------|------|
| gRPC (Indexer ↔ Embedding) | TLS未使用 | ローカル限定 |
| gRPC (MCP ↔ Embedding) | TLS未使用 | ローカル限定 |
| HTTP (→ Qdrant) | 認証キー使用 | 環境変数から取得 |
| stdio (Claude ↔ MCP) | セキュアチャネル | MCP Protocolの仕様 |

### 9.2 認証情報管理

**方針**:
- Qdrant APIキーは環境変数で管理
- 設定ファイルには`${QDRANT_API_KEY}`形式で記載
- ハードコード禁止

**例**:
```yaml
qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"
```

### 9.3 入力検証

**Indexer**:
- ファイルパスの検証（パストラバーサル防止）
- バイナリファイルの除外

**MCP Server**:
- クエリ長の制限
- limit パラメータの範囲チェック（1～50）
- コレクション名の検証

---

## 10. 運用設計

### 10.1 デプロイメント構成

**対応要件ID**: DEP-001～005

| コンポーネント | 配布形態 | デプロイ方法 |
|--------------|---------|------------|
| Indexer | シングルバイナリ | クロスコンパイル（Windows/Linux/macOS） |
| Embedding Server | Dockerイメージ | Docker Compose |
| MCP Server | Dockerイメージ | Docker Compose |
| Qdrant | 公式Dockerイメージ | Docker Compose |

### 10.2 起動順序

1. Qdrant起動
2. Embedding Server起動
3. MCP Server起動（depends_on設定）
4. Claude Desktop設定
5. Indexer実行

### 10.3 ログ設計

#### 10.3.1 Indexerログ

**ログレベル**:
- ERROR: 致命的エラー
- WARN: スキップ可能なエラー
- INFO: 進捗情報
- DEBUG: 詳細情報

**ログ項目**:
- タイムスタンプ
- ログレベル
- メッセージ
- ファイルパス（該当する場合）
- エラースタックトレース（エラー時）

#### 10.3.2 Embedding Serverログ

**ログ項目**:
- リクエスト受信時刻
- モデル名
- テキスト数
- 処理時間
- エラー情報

#### 10.3.3 MCP Serverログ

**ログ項目**:
- ツール呼び出し名
- パラメータ
- 検索結果件数
- 処理時間
- エラー情報

### 10.4 モニタリング

**監視項目**:
- Embedding Server稼働状態（Healthエンドポイント）
- Qdrant稼働状態（HTTPステータス）
- MCP Server応答性（Claude Desktopログ）
- Indexer処理進捗（ログ出力）

---

## 11. 制約事項

### 11.1 技術的制約

**対応要件ID**: CNS-001～009

| 制約ID | 制約内容 | 影響 |
|--------|---------|------|
| CNS-001 | 埋め込みモデルはPython実行 | Goから直接実行不可 |
| CNS-002 | gRPC通信はローカル限定 | TLS未実装 |
| CNS-003 | MCP通信はstdio限定 | ネットワーク経由不可 |
| CNS-004 | Qdrant 1.7.0以上必要 | バージョン確認必須 |
| CNS-005 | Go 1.21以上必要 | ビルド環境制約 |
| CNS-006 | Python 3.11以上必要 | 依存ライブラリ制約 |
| CNS-007 | UTF-8エンコーディング推奨 | 他エンコーディングは未保証 |
| CNS-008 | モデル統一必須 | Indexer/MCP間 |
| CNS-009 | スキーマは docs/data-schema.md | 変更時は両コンポーネント更新 |

### 11.2 運用上の制約

**対応要件ID**: CNS-101～107

| 制約ID | 制約内容 | 対処方法 |
|--------|---------|---------|
| CNS-101 | Embedding Server事前起動必須 | docker-compose使用 |
| CNS-102 | Qdrant事前起動必須 | docker-compose使用 |
| CNS-103 | 初回モデルダウンロード（数GB） | 事前ダウンロード推奨 |
| CNS-104 | コレクション上書き時データ削除 | バックアップ不要の確認 |
| CNS-105 | モデル変更時再登録必要 | 既存データ無効化 |
| CNS-106 | gRPCタイムアウト30秒 | 長時間処理は分割 |
| CNS-107 | Claude Desktop経由のみ | CLI直接実行不可 |

---

## 付録A: データ構造一覧

### A.1 FunctionInfo（Go）

```go
type FunctionInfo struct {
    Name          string   `json:"name"`
    Signature     string   `json:"signature"`
    Parameters    []Param  `json:"parameters"`
    ReturnType    string   `json:"return_type"`
    FilePath      string   `json:"file_path"`
    StartLine     int      `json:"start_line"`
    EndLine       int      `json:"end_line"`
    StartColumn   int      `json:"start_column"`
    EndColumn     int      `json:"end_column"`
    ClassName     string   `json:"class_name,omitempty"`
    ClassHierarchy []string `json:"class_hierarchy,omitempty"`
    Imports       []string `json:"imports"`
    CalledFuncs   []string `json:"called_functions"`
    CalledBy      []string `json:"called_by,omitempty"`
    Docstring     string   `json:"docstring,omitempty"`
    Comments      []string `json:"comments,omitempty"`
    Scope         string   `json:"scope"`
    Modifiers     []string `json:"modifiers"`
    UsedTypes     []string `json:"used_types"`
    Complexity    int      `json:"complexity,omitempty"`
    LOC           int      `json:"loc"`
    CommentLines  int      `json:"comment_lines"`
    Decorators    []string `json:"decorators,omitempty"`
    Annotations   []string `json:"annotations,omitempty"`
    Attributes    []string `json:"attributes,omitempty"`
    Body          string   `json:"body"`
    Language      string   `json:"language"`
}

type Param struct {
    Name string `json:"name"`
    Type string `json:"type,omitempty"`
}
```

### A.2 EmbeddingClient（Go）

```go
type EmbeddingClient struct {
    conn       *grpc.ClientConn
    client     pb.EmbeddingServiceClient
    timeout    time.Duration
    maxRetries int
    batchSize  int
}
```

### A.3 ModelConfig（Python）

```python
@dataclass
class ModelConfig:
    name: str
    dimension: int
    max_length: int
    trust_remote_code: bool
    preload: bool
```

### A.4 SearchResult（Python）

```python
@dataclass
class SearchResult:
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    signature: str
    body: str
    docstring: Optional[str]
    language: str
    complexity: Optional[int]
```

---

## 付録B: 設定ファイル例

### B.1 Indexer設定（config.yaml）

```yaml
input:
  source_dir: "/path/to/source/code"
  ignore_file: ".ragignore"

qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"
  collection_name: "my-project"

embedding_server:
  address: "localhost:50051"
  timeout: 30s
  max_retries: 3
  batch_size: 8
  model_name: "jinaai/jina-embeddings-v2-base-code"

processing:
  parallel_workers: 4
  languages:
    - python
    - rust
    - go
    - java
    - c
    - cpp

logging:
  level: "INFO"
  file: "code-rag-indexer.log"
```

### B.2 Embedding Server設定（embedding-server.yaml）

```yaml
server:
  host: "0.0.0.0"
  port: 50051
  max_workers: 10

default_model: "jinaai/jina-embeddings-v2-base-code"

models:
  - name: "jinaai/jina-embeddings-v2-base-code"
    dimension: 768
    max_length: 8192
    trust_remote_code: true
    preload: true

device: "auto"

logging_level: "INFO"
logging_file: "/var/log/embedding-server.log"
```

### B.3 MCP Server設定（mcp-server.yaml）

```yaml
qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"

embedding_server:
  address: "localhost:50051"
  timeout: 30s
  max_retries: 3
  model_name: "jinaai/jina-embeddings-v2-base-code"

search:
  default_limit: 10
  max_limit: 50
  timeout: 30s

logging:
  level: "INFO"
  file: "/var/log/mcp-server.log"
```

---

## 付録C: 処理シーケンス図

### C.1 登録フローシーケンス

```
Indexer              Embedding Server         Qdrant
   |                        |                    |
   |---(1) Health Check---->|                    |
   |<------- OK ------------|                    |
   |                        |                    |
   |---(2) EmbedBatch ----->|                    |
   |    (codes[])           |                    |
   |                        |--GPU Inference     |
   |<-- vectors[][] --------|                    |
   |                        |                    |
   |---(3) PUT /collections/{name}-------------->|
   |<------------ Collection Created ------------|
   |                        |                    |
   |---(4) POST /collections/{name}/points------>|
   |    (vectors + payloads)|                    |
   |<------------ Points Inserted ---------------|
   |                        |                    |
```

### C.2 検索フローシーケンス

```
Claude      MCP Server      Embedding Server    Qdrant
   |             |                  |              |
   |-(1) search_code-->|            |              |
   |   (query)         |            |              |
   |                   |-(2) EmbedBatch-->|        |
   |                   |<--vector[]-------|        |
   |                   |                  |        |
   |                   |-(3) POST /search--------->|
   |                   |           |               |
   |                   |<--results[]---------------|
   |                   |           |               |
   |<--(4) results[]---|           |               |
   |                   |           |               |
```

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0 | 2025-11-25 | 初版作成 |
