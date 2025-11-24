# Code RAG Indexer - 実装タスクリスト（TDD方式・モノレポ構成）

**開発手法**: テスト駆動開発（TDD - Test-Driven Development）  
**参考**: t_wada氏のTDD原則  
**リポジトリ構成**: モノレポ

---

## TDD基本原則（再確認）

### TDDサイクル

```
1. Red:   失敗するテストを書く
2. Green:  テストが通る最小限の実装
3. Refactor: コードを改善（テストは通ったまま）
```

### 実装ルール

- **仮実装を経て本実装へ**: 最初は定数を返す実装でテストを通し、徐々に一般化
- **明白な実装**: シンプルな場合は直接実装
- **三角測量**: 複数のテストケースから実装を導出
- **1つずつ**: 一度に1つのテストだけを追加

---

## Phase 0: プロジェクトセットアップ

### Task 0-1: リポジトリ初期化

**作業内容**:
```bash
# ディレクトリ構造作成
mkdir -p code-rag-indexer/{indexer/{cmd/indexer,internal/{config,scanner,parser,embedder,indexer,logger},tests/{unit,integration,fixtures}},embedding-server/{server,tests,proto/embedding},proto,scripts,docs}
cd code-rag-indexer
```

**成果物**:
- モノレポ構造
- `.gitignore`
- `README.md`（全体説明）
- `ARCHITECTURE.md`（アーキテクチャ説明）

**.gitignore**:
```
# Go
indexer/bin/
*.exe
*.test
*.out

# Python
embedding-server/__pycache__/
embedding-server/*.pyc
embedding-server/.pytest_cache/
embedding-server/venv/

# Proto生成ファイル（Gitに含めない）
proto/embedding/*.pb.go
embedding-server/proto/embedding/*_pb2.py
embedding-server/proto/embedding/*_pb2_grpc.py

# データ
qdrant_data/
model_cache/
*.log

# IDE
.vscode/
.idea/
```

**完了条件**:
- [ ] ディレクトリ構造作成完了
- [ ] Git初期化完了
- [ ] README作成完了

---

### Task 0-2: Go環境セットアップ

**作業内容**:
```bash
cd indexer
go mod init github.com/yourorg/code-rag-indexer
```

**go.mod初期内容**:
```go
module github.com/yourorg/code-rag-indexer

go 1.21

require (
    github.com/smacker/go-tree-sitter v0.0.0-20231219031718-233c2616a01b
    github.com/qdrant/go-client v1.7.0
    google.golang.org/grpc v1.60.0
    google.golang.org/protobuf v1.31.0
    gopkg.in/yaml.v3 v3.0.1
    github.com/sirupsen/logrus v1.9.3
    github.com/spf13/cobra v1.8.0
    github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00
)
```

**完了条件**:
- [ ] go.mod作成完了
- [ ] 依存パッケージダウンロード完了

---

### Task 0-3: Python環境セットアップ

**作業内容**:
```bash
cd ../embedding-server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**requirements.txt**:
```txt
# gRPC
grpcio>=1.60.0
grpcio-tools>=1.60.0

# ML
transformers>=4.35.0
torch>=2.2.0

# テスト
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.12.0

# ユーティリティ
PyYAML>=6.0
```

**完了条件**:
- [ ] venv作成完了
- [ ] requirements.txtインストール完了

---

### Task 0-4: Protocol Buffers定義とコード生成

#### Step 1: proto定義作成（Red）

**ファイル**: `proto/embedding.proto`

**内容**: 基本設計書のSection 3.1参照

#### Step 2: コード生成スクリプト作成

**Makefile**:
```makefile
.PHONY: proto proto-go proto-python

proto: proto-go proto-python

proto-go:
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       proto/embedding.proto

proto-python:
	cd embedding-server && \
	python -m grpc_tools.protoc -I../proto \
	       --python_out=proto/embedding \
	       --grpc_python_out=proto/embedding \
	       ../proto/embedding.proto
```

#### Step 3: コード生成実行（Green）

```bash
make proto
```

**完了条件**:
- [ ] embedding.proto作成完了
- [ ] Goコード生成成功
- [ ] Pythonコード生成成功
- [ ] 生成ファイルがインポート可能

---

## Phase 1: Python埋め込みサーバー実装

### Task 1-1: 設定読み込み（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `embedding-server/tests/test_config.py`

**テストケース**:
```python
import pytest
from server.config import load_config, Config, ServerConfig, ModelConfig

def test_load_default_config():
    """デフォルト設定が読み込める"""
    config = load_config()
    
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 50051
    assert config.default_model == "jinaai/jina-embeddings-v2-base-code"

def test_load_config_from_file(tmp_path):
    """YAMLファイルから設定を読み込める"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
server:
  host: "127.0.0.1"
  port: 50052

default_model: "microsoft/unixcoder-base"

models:
  - name: "microsoft/unixcoder-base"
    dimension: 768
    max_length: 512
    trust_remote_code: false
    preload: true
""")
    
    config = load_config(str(config_file))
    
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 50052
    assert config.default_model == "microsoft/unixcoder-base"
    assert len(config.models) == 1
    assert config.models[0].preload == True

def test_model_config_validation():
    """モデル設定のバリデーション"""
    model = ModelConfig(
        name="test/model",
        dimension=768,
        max_length=512
    )
    
    assert model.name == "test/model"
    assert model.dimension == 768
    assert model.trust_remote_code == False  # デフォルト値
```

#### Step 2: 仮実装（Green）

**ファイル**: `embedding-server/server/config.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10

@dataclass
class ModelConfig:
    name: str
    dimension: int
    max_length: int
    trust_remote_code: bool = False
    preload: bool = False

@dataclass
class Config:
    server: ServerConfig
    default_model: str
    models: List[ModelConfig]
    device: str = "auto"
    logging_level: str = "INFO"
    logging_file: Optional[str] = None

def load_config(config_path: str = None) -> Config:
    """設定を読み込む（仮実装）"""
    # まずは固定値を返す
    return Config(
        server=ServerConfig(),
        default_model="jinaai/jina-embeddings-v2-base-code",
        models=[
            ModelConfig(
                name="jinaai/jina-embeddings-v2-base-code",
                dimension=768,
                max_length=8192,
                trust_remote_code=True,
                preload=True
            )
        ]
    )
```

#### Step 3: 本実装（Green）

```python
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> Config:
    """設定ファイルを読み込む"""
    if not config_path:
        # デフォルト設定
        return Config(
            server=ServerConfig(),
            default_model="jinaai/jina-embeddings-v2-base-code",
            models=[
                ModelConfig(
                    name="jinaai/jina-embeddings-v2-base-code",
                    dimension=768,
                    max_length=8192,
                    trust_remote_code=True,
                    preload=True
                )
            ]
        )
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # ServerConfig
    server_data = data.get('server', {})
    server = ServerConfig(
        host=server_data.get('host', '0.0.0.0'),
        port=server_data.get('port', 50051),
        max_workers=server_data.get('max_workers', 10)
    )
    
    # ModelConfig
    models = []
    for m in data.get('models', []):
        models.append(ModelConfig(
            name=m['name'],
            dimension=m['dimension'],
            max_length=m['max_length'],
            trust_remote_code=m.get('trust_remote_code', False),
            preload=m.get('preload', False)
        ))
    
    return Config(
        server=server,
        default_model=data.get('default_model', ''),
        models=models,
        device=data.get('device', 'auto'),
        logging_level=data.get('logging_level', 'INFO'),
        logging_file=data.get('logging_file')
    )
```

#### Step 4: リファクタリング（Refactor）

- 関数分割
- エラーハンドリング追加

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 90%

---

### Task 1-2: モデルマネージャー（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `embedding-server/tests/test_model_manager.py`

**テストケース**:
```python
import pytest
from server.model_manager import ModelManager, EmbeddingModel
from server.config import Config, ModelConfig

@pytest.fixture
def test_config():
    """テスト用設定"""
    return Config(
        server=None,
        default_model="test/model",
        models=[
            ModelConfig(
                name="test/model",
                dimension=768,
                max_length=512,
                trust_remote_code=False,
                preload=False
            )
        ],
        device="cpu"  # テストはCPUで
    )

def test_model_manager_initialization(test_config):
    """ModelManagerが初期化できる"""
    manager = ModelManager(test_config)
    
    assert manager.device == "cpu"
    assert len(manager.model_configs) == 1

def test_load_model_lazy(test_config, mocker):
    """モデルが遅延ロードされる"""
    # AutoModelをモック
    mock_model = mocker.patch('server.model_manager.AutoModel.from_pretrained')
    mock_tokenizer = mocker.patch('server.model_manager.AutoTokenizer.from_pretrained')
    
    manager = ModelManager(test_config)
    
    # まだロードされていない
    assert not manager.is_loaded("test/model")
    
    # get_modelでロード
    model = manager.get_model("test/model")
    
    # ロード済み
    assert manager.is_loaded("test/model")
    assert mock_model.called
    assert mock_tokenizer.called

def test_model_cache(test_config, mocker):
    """モデルがキャッシュされる"""
    mock_model = mocker.patch('server.model_manager.AutoModel.from_pretrained')
    
    manager = ModelManager(test_config)
    
    # 1回目
    model1 = manager.get_model("test/model")
    assert mock_model.call_count == 1
    
    # 2回目（キャッシュから取得）
    model2 = manager.get_model("test/model")
    assert mock_model.call_count == 1  # 増えない
    
    # 同じインスタンス
    assert model1 is model2

def test_unsupported_model(test_config):
    """サポート外のモデルでエラー"""
    manager = ModelManager(test_config)
    
    with pytest.raises(ValueError, match="Unsupported model"):
        manager.get_model("unknown/model")

def test_list_loaded_models(test_config, mocker):
    """ロード済みモデル一覧を取得"""
    mocker.patch('server.model_manager.AutoModel.from_pretrained')
    mocker.patch('server.model_manager.AutoTokenizer.from_pretrained')
    
    manager = ModelManager(test_config)
    
    assert manager.list_loaded_models() == []
    
    manager.get_model("test/model")
    
    assert manager.list_loaded_models() == ["test/model"]
```

#### Step 2: 仮実装（Green）

**ファイル**: `embedding-server/server/model_manager.py`

```python
from typing import Dict, List

class EmbeddingModel:
    """埋め込みモデルのラッパー（仮実装）"""
    
    def __init__(self, name: str, config, device: str):
        self.name = name
        self.config = config
        self.device = device
    
    def embed(self, texts: List[str], max_length=None) -> List[List[float]]:
        # 仮実装: ダミーベクトルを返す
        return [[0.0] * self.config.dimension for _ in texts]

class ModelManager:
    """モデルマネージャー（仮実装）"""
    
    def __init__(self, config):
        self.config = config
        self.models: Dict[str, EmbeddingModel] = {}
        self.device = config.device
        self.model_configs = {m.name: m for m in config.models}
    
    def get_model(self, model_name: str) -> EmbeddingModel:
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 仮実装: ダミーモデル
        config = self.model_configs[model_name]
        model = EmbeddingModel(model_name, config, self.device)
        self.models[model_name] = model
        
        return model
    
    def is_loaded(self, model_name: str) -> bool:
        return model_name in self.models
    
    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())
    
    def get_device(self) -> str:
        return self.device
```

#### Step 3: 本実装（Green）

基本設計書のSection 6.3参照

#### Step 4: リファクタリング（Refactor）

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 85%
- [ ] 実際のモデルでの動作確認（手動）

---

### Task 1-3: gRPC Servicer実装（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `embedding-server/tests/test_servicer.py`

**テストケース**:
```python
import pytest
import grpc
from proto.embedding import embedding_pb2, embedding_pb2_grpc
from server.servicer import EmbeddingServicer

@pytest.fixture
def servicer(test_config, mocker):
    """テスト用Servicer"""
    # ModelManagerをモック
    mock_manager = mocker.patch('server.servicer.ModelManager')
    return EmbeddingServicer(test_config)

def test_health_check(servicer):
    """ヘルスチェックが成功する"""
    request = embedding_pb2.HealthRequest()
    context = MockContext()
    
    response = servicer.Health(request, context)
    
    assert response.healthy == True

def test_embed_batch_success(servicer, mocker):
    """バッチ埋め込みが成功する"""
    # モックモデル
    mock_model = mocker.Mock()
    mock_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    servicer.model_manager.get_model.return_value = mock_model
    
    request = embedding_pb2.EmbedBatchRequest(
        texts=["code1", "code2"],
        model_name="test/model"
    )
    context = MockContext()
    
    response = servicer.EmbedBatch(request, context)
    
    assert len(response.vectors) == 2
    assert list(response.vectors[0].values) == [0.1, 0.2, 0.3]
    assert response.model_used == "test/model"

def test_embed_batch_empty_texts(servicer):
    """空のテキストでエラー"""
    request = embedding_pb2.EmbedBatchRequest(texts=[])
    context = MockContext()
    
    response = servicer.EmbedBatch(request, context)
    
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT

def test_embed_batch_unsupported_model(servicer):
    """サポート外のモデルでエラー"""
    servicer.model_manager.get_model.side_effect = ValueError("Unsupported")
    
    request = embedding_pb2.EmbedBatchRequest(
        texts=["code"],
        model_name="unknown/model"
    )
    context = MockContext()
    
    response = servicer.EmbedBatch(request, context)
    
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT

def test_list_models(servicer, test_config):
    """モデル一覧を取得"""
    request = embedding_pb2.ListModelsRequest()
    context = MockContext()
    
    response = servicer.ListModels(request, context)
    
    assert response.default_model == test_config.default_model
    assert len(response.models) == len(test_config.models)

class MockContext:
    """モックgRPCコンテキスト"""
    def __init__(self):
        self.code = None
        self.details = None
    
    def set_code(self, code):
        self.code = code
    
    def set_details(self, details):
        self.details = details
```

#### Step 2-4: 実装

基本設計書のSection 6.2参照

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 85%

---

### Task 1-4: サーバー起動とE2Eテスト

#### Step 1: main.py実装

基本設計書のSection 6.1参照

#### Step 2: E2Eテスト

**ファイル**: `embedding-server/tests/test_e2e.py`

```python
import pytest
import grpc
from proto.embedding import embedding_pb2, embedding_pb2_grpc
import subprocess
import time

@pytest.fixture(scope="module")
def grpc_server():
    """テスト用gRPCサーバーを起動"""
    # サーバープロセス起動
    process = subprocess.Popen([
        "python", "-m", "server.main",
        "--host", "localhost",
        "--port", "50052"
    ])
    
    # 起動待ち
    time.sleep(5)
    
    yield "localhost:50052"
    
    # 終了
    process.terminate()
    process.wait()

def test_e2e_health_check(grpc_server):
    """E2E: ヘルスチェック"""
    channel = grpc.insecure_channel(grpc_server)
    stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
    
    response = stub.Health(embedding_pb2.HealthRequest())
    
    assert response.healthy == True

def test_e2e_embed_batch(grpc_server):
    """E2E: バッチ埋め込み"""
    channel = grpc.insecure_channel(grpc_server)
    stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
    
    request = embedding_pb2.EmbedBatchRequest(
        texts=["def hello(): pass", "def world(): return 42"]
    )
    
    response = stub.EmbedBatch(request)
    
    assert len(response.vectors) == 2
    assert len(response.vectors[0].values) == 768  # Jina dimension
```

**完了条件**:
- [ ] サーバーが起動する
- [ ] E2Eテストがpass
- [ ] 実際のモデルロードが成功

---

## Phase 2: Go側実装（基盤）

### Task 2-1: Config Loader（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/config/loader_test.go`

**テストケース**:
```go
package config

import (
    "os"
    "path/filepath"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestLoad(t *testing.T) {
    t.Run("valid config", func(t *testing.T) {
        // テスト用YAMLファイル作成
        content := `
input:
  source_dir: "/tmp/test"
  ignore_file: ".ragignore"

qdrant:
  url: "http://localhost:6333"
  collection_name: "test-collection"

embedding_server:
  address: "localhost:50051"
  timeout: 30s
  max_retries: 3
  batch_size: 8

processing:
  parallel_workers: 4
  languages:
    - python
    - go

logging:
  level: "INFO"
  file: "test.log"
`
        tmpFile := createTempFile(t, content)
        defer os.Remove(tmpFile)
        
        cfg, err := Load(tmpFile)
        
        require.NoError(t, err)
        assert.Equal(t, "/tmp/test", cfg.Input.SourceDir)
        assert.Equal(t, "test-collection", cfg.Qdrant.CollectionName)
        assert.Equal(t, 30*time.Second, cfg.EmbeddingServer.Timeout)
        assert.Equal(t, 4, cfg.Processing.ParallelWorkers)
        assert.Len(t, cfg.Processing.Languages, 2)
    })
    
    t.Run("file not found", func(t *testing.T) {
        _, err := Load("nonexistent.yaml")
        
        assert.Error(t, err)
    })
    
    t.Run("invalid yaml", func(t *testing.T) {
        content := "invalid: yaml: content:"
        tmpFile := createTempFile(t, content)
        defer os.Remove(tmpFile)
        
        _, err := Load(tmpFile)
        
        assert.Error(t, err)
    })
    
    t.Run("env var expansion", func(t *testing.T) {
        os.Setenv("TEST_API_KEY", "secret123")
        defer os.Unsetenv("TEST_API_KEY")
        
        content := `
qdrant:
  url: "http://localhost:6333"
  api_key: "${TEST_API_KEY}"
  collection_name: "test"
`
        tmpFile := createTempFile(t, content)
        defer os.Remove(tmpFile)
        
        cfg, err := Load(tmpFile)
        
        require.NoError(t, err)
        assert.Equal(t, "secret123", cfg.Qdrant.APIKey)
    })
    
    t.Run("default values", func(t *testing.T) {
        content := `
input:
  source_dir: "/tmp/test"

qdrant:
  url: "http://localhost:6333"
  collection_name: "test"

embedding_server:
  address: "localhost:50051"
`
        tmpFile := createTempFile(t, content)
        defer os.Remove(tmpFile)
        
        cfg, err := Load(tmpFile)
        
        require.NoError(t, err)
        assert.Equal(t, ".ragignore", cfg.Input.IgnoreFile) // デフォルト
        assert.Equal(t, 30*time.Second, cfg.EmbeddingServer.Timeout) // デフォルト
        assert.Equal(t, 3, cfg.EmbeddingServer.MaxRetries) // デフォルト
    })
}

func TestValidate(t *testing.T) {
    t.Run("valid config", func(t *testing.T) {
        // 一時ディレクトリ作成
        tmpDir := t.TempDir()
        
        cfg := &Config{
            Input: InputConfig{
                SourceDir: tmpDir,
            },
            Qdrant: QdrantConfig{
                URL: "http://localhost:6333",
            },
            EmbeddingServer: EmbeddingServerConfig{
                Address: "localhost:50051",
            },
        }
        
        err := validate(cfg)
        
        assert.NoError(t, err)
    })
    
    t.Run("source_dir not exists", func(t *testing.T) {
        cfg := &Config{
            Input: InputConfig{
                SourceDir: "/nonexistent/path",
            },
        }
        
        err := validate(cfg)
        
        assert.Error(t, err)
        assert.Contains(t, err.Error(), "does not exist")
    })
    
    t.Run("invalid qdrant url", func(t *testing.T) {
        tmpDir := t.TempDir()
        
        cfg := &Config{
            Input: InputConfig{
                SourceDir: tmpDir,
            },
            Qdrant: QdrantConfig{
                URL: "invalid-url",
            },
        }
        
        err := validate(cfg)
        
        assert.Error(t, err)
        assert.Contains(t, err.Error(), "Invalid Qdrant URL")
    })
}

func createTempFile(t *testing.T, content string) string {
    tmpFile, err := os.CreateTemp("", "config-*.yaml")
    require.NoError(t, err)
    
    _, err = tmpFile.WriteString(content)
    require.NoError(t, err)
    
    err = tmpFile.Close()
    require.NoError(t, err)
    
    return tmpFile.Name()
}
```

#### Step 2-4: 実装

基本設計書のSection 5.2参照

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 90%

---

### Task 2-2: Logger（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/logger/logger_test.go`

```go
package logger

import (
    "bytes"
    "os"
    "strings"
    "testing"
    
    "github.com/sirupsen/logrus"
    "github.com/stretchr/testify/assert"
)

func TestSetup(t *testing.T) {
    t.Run("console only", func(t *testing.T) {
        // バッファに出力を捕捉
        var buf bytes.Buffer
        
        cfg := LoggingConfig{
            Level: "INFO",
        }
        
        log := Setup(cfg)
        log.Out = &buf
        
        log.Info("test message")
        
        output := buf.String()
        assert.Contains(t, output, "test message")
        assert.Contains(t, output, "INFO")
    })
    
    t.Run("with log file", func(t *testing.T) {
        tmpFile := filepath.Join(t.TempDir(), "test.log")
        
        cfg := LoggingConfig{
            Level: "DEBUG",
            File:  tmpFile,
        }
        
        log := Setup(cfg)
        log.Debug("debug message")
        log.Info("info message")
        
        // ファイルに書き込まれているか確認
        content, err := os.ReadFile(tmpFile)
        assert.NoError(t, err)
        assert.Contains(t, string(content), "debug message")
        assert.Contains(t, string(content), "info message")
    })
    
    t.Run("log levels", func(t *testing.T) {
        var buf bytes.Buffer
        
        cfg := LoggingConfig{
            Level: "WARN",
        }
        
        log := Setup(cfg)
        log.Out = &buf
        
        log.Debug("debug")  // 出力されない
        log.Info("info")    // 出力されない
        log.Warn("warn")    // 出力される
        log.Error("error")  // 出力される
        
        output := buf.String()
        assert.NotContains(t, output, "debug")
        assert.NotContains(t, output, "info")
        assert.Contains(t, output, "warn")
        assert.Contains(t, output, "error")
    })
}

func TestGet(t *testing.T) {
    t.Run("returns logger", func(t *testing.T) {
        Setup(LoggingConfig{Level: "INFO"})
        
        log := Get()
        
        assert.NotNil(t, log)
    })
}
```

#### Step 2-4: 実装

```go
package logger

import (
    "io"
    "os"
    
    "github.com/sirupsen/logrus"
)

var log *logrus.Logger

type LoggingConfig struct {
    Level string
    File  string
}

func Setup(cfg LoggingConfig) *logrus.Logger {
    log = logrus.New()
    
    // ログレベル設定
    level, err := logrus.ParseLevel(cfg.Level)
    if err != nil {
        level = logrus.InfoLevel
    }
    log.SetLevel(level)
    
    // フォーマット設定
    log.SetFormatter(&logrus.TextFormatter{
        FullTimestamp: true,
        TimestampFormat: "2006-01-02 15:04:05",
    })
    
    // 出力先設定
    var writers []io.Writer
    writers = append(writers, os.Stdout) // コンソール
    
    if cfg.File != "" {
        file, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
        if err == nil {
            writers = append(writers, file)
        }
    }
    
    log.SetOutput(io.MultiWriter(writers...))
    
    return log
}

func Get() *logrus.Logger {
    if log == nil {
        log = logrus.New()
    }
    return log
}
```

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 90%

---

### Task 2-3: File Scanner（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/scanner/scanner_test.go`

```go
package scanner

import (
    "os"
    "path/filepath"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestScan(t *testing.T) {
    t.Run("scan python files", func(t *testing.T) {
        // テスト用ディレクトリ構造作成
        tmpDir := setupTestDir(t, map[string]string{
            "main.py":         "print('hello')",
            "lib.py":          "def func(): pass",
            "subdir/test.py":  "import unittest",
            "readme.md":       "# README",
        })
        
        scanner := New(InputConfig{
            SourceDir: tmpDir,
        }, []string{"python"})
        
        files, err := scanner.Scan()
        
        require.NoError(t, err)
        assert.Len(t, files, 3) // .pyファイルのみ
        assert.Contains(t, filenames(files), "main.py")
        assert.Contains(t, filenames(files), "lib.py")
        assert.Contains(t, filenames(files), filepath.Join("subdir", "test.py"))
    })
    
    t.Run("with .ragignore", func(t *testing.T) {
        tmpDir := setupTestDir(t, map[string]string{
            "main.py":            "code",
            "test_main.py":       "test",
            "build/generated.py": "generated",
            ".ragignore":         "test_*.py\nbuild/\n",
        })
        
        scanner := New(InputConfig{
            SourceDir: tmpDir,
            IgnoreFile: ".ragignore",
        }, []string{"python"})
        
        files, err := scanner.Scan()
        
        require.NoError(t, err)
        assert.Len(t, files, 1) // main.pyのみ
        assert.Contains(t, filenames(files), "main.py")
    })
    
    t.Run("exclude binary files", func(t *testing.T) {
        tmpDir := setupTestDir(t, map[string]string{
            "code.py":    "print('hello')",
            "binary.exe": "\x00\x01\x02binary",
        })
        
        scanner := New(InputConfig{
            SourceDir: tmpDir,
        }, []string{"python"})
        
        files, err := scanner.Scan()
        
        require.NoError(t, err)
        assert.Len(t, files, 1)
        assert.Contains(t, filenames(files), "code.py")
    })
    
    t.Run("multiple languages", func(t *testing.T) {
        tmpDir := setupTestDir(t, map[string]string{
            "main.py":   "python",
            "lib.rs":    "rust",
            "app.go":    "golang",
            "Main.java": "java",
            "util.c":    "c code",
        })
        
        scanner := New(InputConfig{
            SourceDir: tmpDir,
        }, []string{"python", "rust", "go"})
        
        files, err := scanner.Scan()
        
        require.NoError(t, err)
        assert.Len(t, files, 3) // python, rust, goのみ
    })
}

func setupTestDir(t *testing.T, files map[string]string) string {
    tmpDir := t.TempDir()
    
    for path, content := range files {
        fullPath := filepath.Join(tmpDir, path)
        dir := filepath.Dir(fullPath)
        
        err := os.MkdirAll(dir, 0755)
        require.NoError(t, err)
        
        err = os.WriteFile(fullPath, []byte(content), 0644)
        require.NoError(t, err)
    }
    
    return tmpDir
}

func filenames(paths []string) []string {
    names := make([]string, len(paths))
    for i, p := range paths {
        names[i] = filepath.Base(p)
    }
    return names
}
```

#### Step 2-4: 実装

基本設計書のSection 5.3参照

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 85%

---

## Phase 3: Go側実装（パーサー）

### Task 3-1: Parser Interface & FunctionInfo（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/parser/parser_test.go`

```go
package parser

import (
    "testing"
    
    "github.com/stretchr/testify/assert"
)

func TestFunctionInfo(t *testing.T) {
    t.Run("create function info", func(t *testing.T) {
        info := &FunctionInfo{
            Name:      "testFunc",
            Code:      "def testFunc(): pass",
            FilePath:  "/tmp/test.py",
            StartLine: 1,
            EndLine:   1,
            Language:  "python",
        }
        
        assert.Equal(t, "testFunc", info.Name)
        assert.Equal(t, "python", info.Language)
    })
}

func TestFactory(t *testing.T) {
    t.Run("get python parser", func(t *testing.T) {
        factory := NewFactory()
        
        parser := factory.GetParser("test.py")
        
        assert.NotNil(t, parser)
        assert.Equal(t, "python", parser.Language())
    })
    
    t.Run("get rust parser", func(t *testing.T) {
        factory := NewFactory()
        
        parser := factory.GetParser("main.rs")
        
        assert.NotNil(t, parser)
        assert.Equal(t, "rust", parser.Language())
    })
    
    t.Run("unsupported extension", func(t *testing.T) {
        factory := NewFactory()
        
        parser := factory.GetParser("readme.md")
        
        assert.Nil(t, parser)
    })
}
```

#### Step 2-4: 実装

基本設計書のSection 5.4参照

**完了条件**:
- [ ] 全テストがpass
- [ ] インターフェース定義完了

---

### Task 3-2: Python Parser（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/parser/python_parser_test.go`

```go
package parser

import (
    "os"
    "path/filepath"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestPythonParser_ParseFile(t *testing.T) {
    parser := NewPythonParser()
    
    t.Run("simple function", func(t *testing.T) {
        code := `def hello():
    print("hello")
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        require.NoError(t, err)
        assert.Len(t, functions, 1)
        assert.Equal(t, "hello", functions[0].Name)
        assert.Equal(t, "python", functions[0].Language)
        assert.Equal(t, 1, functions[0].StartLine)
    })
    
    t.Run("function with arguments", func(t *testing.T) {
        code := `def add(a, b):
    return a + b
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        require.NoError(t, err)
        assert.Len(t, functions, 1)
        assert.Equal(t, "add", functions[0].Name)
        assert.ElementsMatch(t, []string{"a", "b"}, functions[0].Arguments)
    })
    
    t.Run("function with docstring", func(t *testing.T) {
        code := `def greet(name):
    """Say hello to someone"""
    print(f"Hello, {name}")
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        require.NoError(t, err)
        assert.Len(t, functions, 1)
        assert.Contains(t, functions[0].Docstring, "Say hello")
    })
    
    t.Run("multiple functions", func(t *testing.T) {
        code := `def func1():
    pass

def func2():
    pass

def func3():
    pass
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        require.NoError(t, err)
        assert.Len(t, functions, 3)
        assert.Equal(t, "func1", functions[0].Name)
        assert.Equal(t, "func2", functions[1].Name)
        assert.Equal(t, "func3", functions[2].Name)
    })
    
    t.Run("class with methods", func(t *testing.T) {
        code := `class MyClass:
    def method1(self):
        pass
    
    def method2(self, arg):
        return arg
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        require.NoError(t, err)
        assert.Len(t, functions, 2)
        assert.Equal(t, "method1", functions[0].Name)
        assert.Equal(t, "method2", functions[1].Name)
    })
    
    t.Run("syntax error", func(t *testing.T) {
        code := `def broken(
    # 構文エラー
`
        tmpFile := createCodeFile(t, code, ".py")
        
        functions, err := parser.ParseFile(tmpFile)
        
        // エラーは返さないが、空リスト
        require.NoError(t, err)
        assert.Len(t, functions, 0)
    })
}

func createCodeFile(t *testing.T, code string, ext string) string {
    tmpFile, err := os.CreateTemp("", "code-*"+ext)
    require.NoError(t, err)
    
    _, err = tmpFile.WriteString(code)
    require.NoError(t, err)
    
    err = tmpFile.Close()
    require.NoError(t, err)
    
    return tmpFile.Name()
}
```

#### Step 2-4: 実装

基本設計書のSection 5.5参照

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 80%
- [ ] 実際のPythonファイルでテスト

---

### Task 3-3~3-7: 他言語パーサー実装

各言語で同様のTDDサイクル:
- Task 3-3: Rust Parser
- Task 3-4: Go Parser
- Task 3-5: Java Parser
- Task 3-6: C Parser
- Task 3-7: C++ Parser

**注**: これらは並行して実装可能

**完了条件（各言語）**:
- [ ] 全テストがpass
- [ ] カバレッジ > 75%

---

## Phase 4: Go側実装（gRPCクライアント）

### Task 4-1: Embedding Client（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/embedder/client_test.go`

```go
package embedder

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "google.golang.org/grpc"
    "google.golang.org/grpc/test/bufconn"
    
    pb "github.com/yourorg/code-rag-indexer/proto/embedding"
)

func TestNewClient(t *testing.T) {
    t.Run("successful connection", func(t *testing.T) {
        // モックgRPCサーバーを起動
        server, addr := startMockServer(t)
        defer server.Stop()
        
        cfg := EmbeddingServerConfig{
            Address:    addr,
            Timeout:    5 * time.Second,
            MaxRetries: 3,
            BatchSize:  8,
        }
        
        client, err := NewClient(cfg)
        
        require.NoError(t, err)
        assert.NotNil(t, client)
        
        client.Close()
    })
    
    t.Run("connection failure", func(t *testing.T) {
        cfg := EmbeddingServerConfig{
            Address: "localhost:99999", // 存在しないポート
            Timeout: 1 * time.Second,
        }
        
        _, err := NewClient(cfg)
        
        assert.Error(t, err)
    })
}

func TestHealthCheck(t *testing.T) {
    t.Run("healthy server", func(t *testing.T) {
        server, addr := startMockServer(t)
        defer server.Stop()
        
        client, _ := NewClient(EmbeddingServerConfig{Address: addr})
        defer client.Close()
        
        err := client.HealthCheck()
        
        assert.NoError(t, err)
    })
}

func TestEmbedBatch(t *testing.T) {
    t.Run("successful embedding", func(t *testing.T) {
        server, addr := startMockServer(t)
        defer server.Stop()
        
        client, _ := NewClient(EmbeddingServerConfig{
            Address:   addr,
            BatchSize: 2,
        })
        defer client.Close()
        
        codes := []string{"code1", "code2"}
        
        vectors, err := client.EmbedBatch(codes)
        
        require.NoError(t, err)
        assert.Len(t, vectors, 2)
        assert.Len(t, vectors[0], 768) // dimension
    })
    
    t.Run("empty input", func(t *testing.T) {
        server, addr := startMockServer(t)
        defer server.Stop()
        
        client, _ := NewClient(EmbeddingServerConfig{Address: addr})
        defer client.Close()
        
        vectors, err := client.EmbedBatch([]string{})
        
        assert.Error(t, err)
        assert.Nil(t, vectors)
    })
}

func startMockServer(t *testing.T) (*grpc.Server, string) {
    // モックサーバー実装（簡易版）
    lis := bufconn.Listen(1024 * 1024)
    server := grpc.NewServer()
    
    pb.RegisterEmbeddingServiceServer(server, &mockEmbeddingServer{})
    
    go func() {
        server.Serve(lis)
    }()
    
    return server, lis.Addr().String()
}

type mockEmbeddingServer struct {
    pb.UnimplementedEmbeddingServiceServer
}

func (s *mockEmbeddingServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
    return &pb.HealthResponse{
        Healthy: true,
        Device:  "cpu",
    }, nil
}

func (s *mockEmbeddingServer) EmbedBatch(ctx context.Context, req *pb.EmbedBatchRequest) (*pb.EmbedBatchResponse, error) {
    resp := &pb.EmbedBatchResponse{
        ModelUsed: "test/model",
    }
    
    for range req.Texts {
        vector := &pb.Vector{
            Values: make([]float32, 768),
        }
        resp.Vectors = append(resp.Vectors, vector)
    }
    
    return resp, nil
}
```

#### Step 2-4: 実装

基本設計書のSection 5.6参照

**完了条件**:
- [ ] 全テストがpass
- [ ] カバレッジ > 85%
- [ ] 実際の埋め込みサーバーでテスト

---

## Phase 5: Go側実装（Qdrant）

### Task 5-1: Qdrant Indexer（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/internal/indexer/qdrant_test.go`

```go
package indexer

import (
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/yourorg/code-rag-indexer/internal/parser"
)

func TestQdrantIndexer(t *testing.T) {
    // Qdrantモックサーバーまたはテスト用Qdrantインスタンスが必要
    
    t.Run("create collection", func(t *testing.T) {
        // テストは統合テストで実施
        t.Skip("Requires Qdrant instance")
    })
    
    t.Run("upsert batch", func(t *testing.T) {
        t.Skip("Requires Qdrant instance")
    })
}

// 統合テストは別ファイルで実施
```

#### Step 2-4: 実装

基本設計書のSection 5.7参照

**完了条件**:
- [ ] 実装完了
- [ ] 統合テストでQdrant動作確認

---

## Phase 6: Go側実装（統合）

### Task 6-1: Main Orchestrator（TDD）

#### Step 1: テスト作成（Red）

**ファイル**: `indexer/cmd/indexer/main_test.go`

```go
package main

import (
    "os"
    "testing"
    
    "github.com/stretchr/testify/assert"
)

func TestParseArguments(t *testing.T) {
    t.Run("with config file", func(t *testing.T) {
        os.Args = []string{"cmd", "-c", "config.yaml"}
        
        // 引数解析テスト
        // （実装後に追加）
    })
}

// メイン処理のテストは統合テストで実施
```

#### Step 2-4: 実装

基本設計書のSection 5.1参照

**完了条件**:
- [ ] CLIが動作する
- [ ] ヘルプが表示される
- [ ] バージョンが表示される

---

## Phase 7: 統合テスト

### Task 7-1: E2Eテスト

**ファイル**: `indexer/tests/integration/e2e_test.go`

```go
package integration

import (
    "os"
    "os/exec"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestE2E(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    // 前提条件チェック
    requireDockerServices(t)
    
    t.Run("full indexing flow", func(t *testing.T) {
        // テスト用ソースコードディレクトリ作成
        srcDir := setupTestSource(t)
        
        // 設定ファイル作成
        configFile := createTestConfig(t, srcDir)
        
        // インデクサー実行
        cmd := exec.Command("./bin/code-rag-indexer", "-c", configFile)
        output, err := cmd.CombinedOutput()
        
        require.NoError(t, err, "Indexer failed: %s", output)
        
        // Qdrantにデータが登録されているか確認
        // （Qdrant APIを使って確認）
        
        assert.Contains(t, string(output), "Indexing completed")
    })
}

func requireDockerServices(t *testing.T) {
    // Qdrant確認
    // 埋め込みサーバー確認
}

func setupTestSource(t *testing.T) string {
    tmpDir := t.TempDir()
    
    // テスト用Pythonファイル作成
    os.WriteFile(
        filepath.Join(tmpDir, "test.py"),
        []byte("def hello(): print('hello')"),
        0644,
    )
    
    return tmpDir
}

func createTestConfig(t *testing.T, srcDir string) string {
    config := fmt.Sprintf(`
input:
  source_dir: %s

qdrant:
  url: "http://localhost:6333"
  collection_name: "test-e2e"

embedding_server:
  address: "localhost:50051"

processing:
  parallel_workers: 2
  languages:
    - python

logging:
  level: "DEBUG"
`, srcDir)
    
    tmpFile := filepath.Join(t.TempDir(), "config.yaml")
    os.WriteFile(tmpFile, []byte(config), 0644)
    
    return tmpFile
}
```

**完了条件**:
- [ ] E2Eテストがpass
- [ ] 実際のソースコードで動作確認

---

## Phase 8: ビルド・デプロイ

### Task 8-1: ビルドスクリプト作成

**ファイル**: `scripts/build.sh`

基本設計書のSection 3.2参照

**完了条件**:
- [ ] 全プラットフォーム向けビルド成功
- [ ] バイナリサイズが適切

---

### Task 8-2: Docker化（埋め込みサーバー）

**ファイル**: `embedding-server/Dockerfile`

基本設計書のSection 3.3参照

**完了条件**:
- [ ] Dockerイメージビルド成功
- [ ] コンテナ起動成功
- [ ] ヘルスチェック成功

---

### Task 8-3: docker-compose.yml作成

基本設計書のSection 3.4参照

**完了条件**:
- [ ] docker-compose up成功
- [ ] 全サービス起動確認

---

## Phase 9: ドキュメント

### Task 9-1: README作成

**ファイル**: `README.md`

**内容**:
- プロジェクト概要
- インストール方法
- 使用方法
- 設定例
- トラブルシューティング

**完了条件**:
- [ ] README完成
- [ ] 手順通りにインストール可能

---

### Task 9-2: API仕様書作成

**ファイル**: `docs/api.md`

**内容**:
- gRPC API仕様
- リクエスト/レスポンス例

**完了条件**:
- [ ] API仕様書完成

---

## 実装順序サマリー

```
Phase 0: セットアップ (4タスク)
  ├─ Task 0-1: リポジトリ初期化
  ├─ Task 0-2: Go環境
  ├─ Task 0-3: Python環境
  └─ Task 0-4: Proto定義

Phase 1: Python埋め込みサーバー (4タスク)
  ├─ Task 1-1: Config
  ├─ Task 1-2: ModelManager
  ├─ Task 1-3: gRPC Servicer
  └─ Task 1-4: Server起動・E2E

Phase 2: Go基盤 (3タスク)
  ├─ Task 2-1: Config Loader
  ├─ Task 2-2: Logger
  └─ Task 2-3: File Scanner

Phase 3: Goパーサー (7タスク) ← 並行可能
  ├─ Task 3-1: Interface
  ├─ Task 3-2: Python Parser
  ├─ Task 3-3: Rust Parser
  ├─ Task 3-4: Go Parser
  ├─ Task 3-5: Java Parser
  ├─ Task 3-6: C Parser
  └─ Task 3-7: C++ Parser

Phase 4: Go gRPCクライアント (1タスク)
  └─ Task 4-1: Embedding Client

Phase 5: Go Qdrant (1タスク)
  └─ Task 5-1: Qdrant Indexer

Phase 6: Go統合 (1タスク)
  └─ Task 6-1: Main Orchestrator

Phase 7: 統合テスト (1タスク)
  └─ Task 7-1: E2Eテスト

Phase 8: ビルド・デプロイ (3タスク)
  ├─ Task 8-1: ビルドスクリプト
  ├─ Task 8-2: Docker化
  └─ Task 8-3: docker-compose

Phase 9: ドキュメント (2タスク)
  ├─ Task 9-1: README
  └─ Task 9-2: API仕様書
```

**総タスク数**: 27タスク  
**推定工数**: 2-3週間（1人での実装）

---

## Claude Codeへの実装依頼方法

各Phaseを順番に実装依頼する際の指示例：

```
【Phase 0: プロジェクトセットアップ】

以下のTDD方式で実装してください：

Task 0-1: リポジトリ初期化
- モノレポ構造を作成
- .gitignore作成
- README.md（基本構成のみ）作成

Task 0-2: Go環境セットアップ
- go.mod初期化
- 依存パッケージ追加

Task 0-3: Python環境セットアップ
- requirements.txt作成
- venv作成手順をREADMEに追記

Task 0-4: Protocol Buffers定義
1. proto/embedding.proto作成
2. Makefile作成（proto生成ターゲット）
3. コード生成実行
4. 生成ファイルの動作確認

完了条件:
- すべてのディレクトリが作成されている
- Protoコードが生成されている
- READMEにセットアップ手順が記載されている
```
