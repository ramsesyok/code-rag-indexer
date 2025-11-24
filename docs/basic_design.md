# Code RAG Indexer - 基本設計書（Go言語版 + gRPC埋め込みサーバー）

**文書バージョン**: 2.1  
**作成日**: 2025-11-23  
**更新日**: 2025-11-23（Docker化の修正）

---

## 1. 配布・デプロイメント方針

### 1.1 配布形態

| コンポーネント | 配布形態 | 理由 |
|--------------|---------|------|
| **Go Indexer** | シングルバイナリ | クロスコンパイル可能、依存なし |
| **Python埋め込みサーバー** | Dockerイメージ | Python依存、モデルキャッシュ管理 |
| **Qdrant** | Dockerイメージ（公式） | データベースサーバー |

### 1.2 デプロイメント構成

```
開発マシン/CI環境
├── code-rag-indexer (シングルバイナリ)
│   ├── Windows: code-rag-indexer.exe
│   ├── Linux: code-rag-indexer
│   └── macOS: code-rag-indexer
│
└── Docker環境
    ├── embedding-server (コンテナ)
    └── qdrant (コンテナ)
```

---

## 2. プロジェクト構造

```
code-rag-indexer/
├── cmd/
│   └── indexer/
│       └── main.go
├── internal/
│   ├── config/
│   ├── scanner/
│   ├── parser/
│   ├── embedder/
│   ├── indexer/
│   └── logger/
├── proto/
│   ├── embedding.proto
│   └── embedding/
│       ├── embedding.pb.go
│       └── embedding_grpc.pb.go
├── embedding-server/              # Python埋め込みサーバー（サブディレクトリ）
│   ├── server/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── servicer.py
│   │   ├── model_manager.py
│   │   └── config.py
│   ├── proto/
│   │   └── embedding/
│   │       ├── embedding_pb2.py
│   │       └── embedding_pb2_grpc.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile                 # ← Python側のみDocker化
│   ├── embedding-server.yaml.example
│   └── README.md
├── tests/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   ├── build.sh                   # クロスコンパイルスクリプト
│   └── install.sh                 # インストールスクリプト
├── go.mod
├── go.sum
├── Makefile
├── docker-compose.yml             # 埋め込みサーバー + Qdrant のみ
├── config.yaml.example
├── .ragignore.example
└── README.md
```

---

## 3. ビルド・デプロイ

### 3.1 Makefile

```makefile
.PHONY: all proto build build-all test clean docker-build docker-up docker-down install

# デフォルトターゲット
all: proto build

# Protocol Buffers生成
proto:
	@echo "Generating Protocol Buffers..."
	# Go
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       proto/embedding.proto
	# Python
	cd embedding-server && \
	python -m grpc_tools.protoc -I../proto \
	       --python_out=proto/embedding \
	       --grpc_python_out=proto/embedding \
	       ../proto/embedding.proto

# ローカル環境用ビルド
build:
	@echo "Building for local platform..."
	go build -o bin/code-rag-indexer cmd/indexer/main.go

# 全プラットフォーム向けビルド（クロスコンパイル）
build-all:
	@echo "Building for all platforms..."
	./scripts/build.sh

# テスト
test:
	@echo "Running tests..."
	go test ./... -v -cover

# Dockerイメージビルド（埋め込みサーバーのみ）
docker-build:
	@echo "Building embedding server Docker image..."
	docker build -t embedding-server:latest embedding-server/

# Docker環境起動（埋め込みサーバー + Qdrant）
docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

# Docker環境停止
docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

# インストール（バイナリを/usr/local/binにコピー）
install: build
	@echo "Installing code-rag-indexer..."
	sudo cp bin/code-rag-indexer /usr/local/bin/
	@echo "Installed to /usr/local/bin/code-rag-indexer"

# クリーンアップ
clean:
	rm -rf bin/
	rm -f proto/embedding/*.pb.go
	rm -f embedding-server/proto/embedding/*_pb2.py
```

### 3.2 クロスコンパイルスクリプト

**ファイル**: `scripts/build.sh`

```bash
#!/bin/bash

set -e

VERSION=${VERSION:-"v1.0.0"}
OUTPUT_DIR="bin"

mkdir -p ${OUTPUT_DIR}

echo "Building code-rag-indexer ${VERSION}..."

# Linux (amd64)
echo "Building for Linux (amd64)..."
GOOS=linux GOARCH=amd64 go build -o ${OUTPUT_DIR}/code-rag-indexer-linux-amd64 \
    -ldflags "-X main.Version=${VERSION}" \
    cmd/indexer/main.go

# Linux (arm64)
echo "Building for Linux (arm64)..."
GOOS=linux GOARCH=arm64 go build -o ${OUTPUT_DIR}/code-rag-indexer-linux-arm64 \
    -ldflags "-X main.Version=${VERSION}" \
    cmd/indexer/main.go

# Windows (amd64)
echo "Building for Windows (amd64)..."
GOOS=windows GOARCH=amd64 go build -o ${OUTPUT_DIR}/code-rag-indexer-windows-amd64.exe \
    -ldflags "-X main.Version=${VERSION}" \
    cmd/indexer/main.go

# macOS (amd64)
echo "Building for macOS (amd64)..."
GOOS=darwin GOARCH=amd64 go build -o ${OUTPUT_DIR}/code-rag-indexer-darwin-amd64 \
    -ldflags "-X main.Version=${VERSION}" \
    cmd/indexer/main.go

# macOS (arm64 - Apple Silicon)
echo "Building for macOS (arm64)..."
GOOS=darwin GOARCH=arm64 go build -o ${OUTPUT_DIR}/code-rag-indexer-darwin-arm64 \
    -ldflags "-X main.Version=${VERSION}" \
    cmd/indexer/main.go

echo "Build complete! Binaries are in ${OUTPUT_DIR}/"
ls -lh ${OUTPUT_DIR}/
```

### 3.3 Dockerfile（Python埋め込みサーバーのみ）

**ファイル**: `embedding-server/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 依存パッケージインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションをコピー
COPY server/ ./server/
COPY proto/ ./proto/

# gRPCポート公開
EXPOSE 50051

# ヘルスチェック
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import grpc; grpc.channel_ready_future(grpc.insecure_channel('localhost:50051')).result(timeout=5)"

# 実行
ENTRYPOINT ["python", "-m", "server.main"]
CMD ["--config", "/app/embedding-server.yaml"]
```

### 3.4 docker-compose.yml

```yaml
version: '3.8'

services:
  # Qdrantサーバー
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"  # gRPCポート（オプション）
    volumes:
      - ./qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # Python埋め込みサーバー
  embedding-server:
    build:
      context: ./embedding-server
      dockerfile: Dockerfile
    container_name: embedding-server
    ports:
      - "50051:50051"
    volumes:
      - ./model_cache:/root/.cache/huggingface
      - ./embedding-server.yaml:/app/embedding-server.yaml:ro
    environment:
      - LOG_LEVEL=INFO
    depends_on:
      - qdrant
    restart: unless-stopped
```

**注意**: Indexerはコンテナ化しないため、docker-compose.ymlから削除

---

## 4. インストール・セットアップ手順

### 4.1 開発環境セットアップ

```bash
# 1. リポジトリクローン
git clone https://github.com/yourorg/code-rag-indexer.git
cd code-rag-indexer

# 2. Go依存関係インストール
go mod download

# 3. Python依存関係インストール（埋め込みサーバー開発時のみ）
cd embedding-server
pip install -r requirements.txt
cd ..

# 4. Protocol Buffers生成
make proto

# 5. ビルド
make build

# 6. Docker環境起動（埋め込みサーバー + Qdrant）
make docker-up

# 7. 動作確認
./bin/code-rag-indexer -c config.yaml
```

### 4.2 本番環境セットアップ

#### Step 1: Docker環境の準備

```bash
# docker-compose.ymlと設定ファイルを配置
mkdir -p /opt/code-rag-indexer
cd /opt/code-rag-indexer

# ファイルを配置
# - docker-compose.yml
# - embedding-server.yaml
# - model_cache/ (ディレクトリ作成)
# - qdrant_data/ (ディレクトリ作成)

# Docker起動
docker-compose up -d

# 起動確認
docker-compose ps
docker-compose logs embedding-server
```

#### Step 2: Goバイナリのインストール

**Linux/macOS:**
```bash
# バイナリをダウンロード（またはビルド）
curl -L https://github.com/yourorg/code-rag-indexer/releases/download/v1.0.0/code-rag-indexer-linux-amd64 -o code-rag-indexer

# 実行権限付与
chmod +x code-rag-indexer

# システムにインストール
sudo mv code-rag-indexer /usr/local/bin/

# 動作確認
code-rag-indexer --version
```

**Windows:**
```powershell
# バイナリをダウンロード
Invoke-WebRequest -Uri "https://github.com/yourorg/code-rag-indexer/releases/download/v1.0.0/code-rag-indexer-windows-amd64.exe" -OutFile "code-rag-indexer.exe"

# PATHに追加（またはカレントディレクトリで実行）
.\code-rag-indexer.exe --version
```

#### Step 3: 設定ファイルの準備

```bash
# config.yamlを作成
cat > config.yaml <<EOF
input:
  source_dir: "/path/to/your/source/code"
  ignore_file: ".ragignore"

qdrant:
  url: "http://localhost:6333"
  api_key: ""
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
EOF
```

#### Step 4: 実行

```bash
# 実行
code-rag-indexer -c config.yaml

# または詳細ログ
code-rag-indexer -c config.yaml -v
```

---

## 5. リリース・配布

### 5.1 GitHub Releasesでの配布

**リリースファイル構成:**
```
code-rag-indexer-v1.0.0/
├── code-rag-indexer-linux-amd64
├── code-rag-indexer-linux-arm64
├── code-rag-indexer-windows-amd64.exe
├── code-rag-indexer-darwin-amd64
├── code-rag-indexer-darwin-arm64
├── config.yaml.example
├── .ragignore.example
├── docker-compose.yml
├── embedding-server.yaml.example
├── INSTALL.md
└── README.md
```

### 5.2 インストールスクリプト

**ファイル**: `scripts/install.sh`

```bash
#!/bin/bash

set -e

VERSION=${VERSION:-"latest"}
INSTALL_DIR=${INSTALL_DIR:-"/usr/local/bin"}

echo "Installing code-rag-indexer..."

# OSとアーキテクチャを検出
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case $ARCH in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

BINARY_NAME="code-rag-indexer-${OS}-${ARCH}"

if [ "$OS" = "windows" ]; then
    BINARY_NAME="${BINARY_NAME}.exe"
fi

# ダウンロード
DOWNLOAD_URL="https://github.com/yourorg/code-rag-indexer/releases/download/${VERSION}/${BINARY_NAME}"

echo "Downloading from: $DOWNLOAD_URL"
curl -L -o code-rag-indexer $DOWNLOAD_URL

# インストール
chmod +x code-rag-indexer
sudo mv code-rag-indexer $INSTALL_DIR/

echo "Successfully installed to $INSTALL_DIR/code-rag-indexer"
echo "Run 'code-rag-indexer --version' to verify installation"
```

**使用方法:**
```bash
curl -sSL https://raw.githubusercontent.com/yourorg/code-rag-indexer/main/scripts/install.sh | bash
```

---

## 6. systemdサービス化（Linux）

埋め込みサーバーをsystemdで管理する場合：

**ファイル**: `/etc/systemd/system/embedding-server.service`

```ini
[Unit]
Description=Code RAG Embedding Server
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/code-rag-indexer
ExecStart=/usr/bin/docker-compose up -d embedding-server
ExecStop=/usr/bin/docker-compose stop embedding-server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**有効化:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable embedding-server
sudo systemctl start embedding-server
sudo systemctl status embedding-server
```
