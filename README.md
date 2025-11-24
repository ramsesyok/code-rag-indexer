# Code RAG Indexer

**ローカルLLM + ローカルRAG環境向けのソースコード検索基盤**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)

---

## 📖 概要

Code RAG Indexerは、大規模なソースコードベースを効率的に検索可能にするためのインデックス登録ツールです。AST（抽象構文木）解析に基づいて関数・メソッド単位でコードを分割し、コード特化型の埋め込みモデルでベクトル化してQdrantに登録します。

### 主な特徴

- **🎯 高精度なコード理解**: AST解析により構文構造を正確に把握
- **🚀 高速処理**: Go言語による並列処理で大規模コードベースにも対応
- **🔌 モジュラーアーキテクチャ**: gRPC経由で埋め込みモデルを分離、柔軟な構成が可能
- **📦 シンプルなデプロイ**: Goバイナリはシングルバイナリ、埋め込みサーバーはDocker化
- **🌐 多言語対応**: Python, Rust, Go, Java, C, C++に対応

---

## 🎯 目的

### 解決する課題

- **大規模コードベースの理解困難**: 数十万～数百万行のコードから必要な箇所を見つけるのは困難
- **コンテキストの欠如**: LLMに全コードを渡すことは不可能、関連コードのみを効率的に取得したい
- **コード特化検索の必要性**: 一般的なテキスト検索では、プログラミング特有の構造や意味を捉えきれない

### このツールができること

1. **ソースコードのインデックス化**
   - ローカルディレクトリまたはGitリポジトリからコードを読み込み
   - AST解析で関数・メソッド・クラス単位に分割
   - メタデータ（引数、戻り値、依存関係等）を抽出

2. **高品質なベクトル化**
   - コード特化型埋め込みモデル（Jina Embeddings v2 Code）を使用
   - 最大8192トークンの長いコンテキストに対応
   - 関数全体を1チャンクとして処理可能

3. **効率的な検索基盤の構築**
   - Qdrantベクトルデータベースに登録
   - 意味的類似性に基づく高速検索が可能
   - リポジトリごとにコレクションを分離管理

---

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────┐
│   Go Indexer (シングルバイナリ)     │
│   - ファイルスキャン                │
│   - AST解析（Tree-sitter）          │
│   - メタデータ抽出                  │
│   - 並列処理                        │
└──────────────┬──────────────────────┘
               │ gRPC
               v
┌─────────────────────────────────────┐
│   Python 埋め込みサーバー（Docker） │
│   - Jina Embeddings v2 Code         │
│   - 複数モデル対応                  │
│   - GPU/CPU自動切り替え             │
└──────────────┬──────────────────────┘
               │
               v
┌─────────────────────────────────────┐
│   Qdrant Vector Database（Docker） │
│   - ベクトル検索                    │
│   - メタデータフィルタリング        │
│   - コレクション管理                │
└─────────────────────────────────────┘
```

### 技術スタック

| コンポーネント | 技術 | 役割 |
|--------------|------|------|
| **Indexer** | Go 1.21+ | メイン処理、AST解析、並列化 |
| **Parser** | Tree-sitter | 多言語対応のAST解析 |
| **Embedding Server** | Python 3.11+, gRPC | コードのベクトル化 |
| **Embedding Model** | Jina Embeddings v2 Code | 768次元、8192トークン対応 |
| **Vector DB** | Qdrant 1.7+ | ベクトルストレージ・検索 |
| **Communication** | gRPC, Protocol Buffers | 高速な型安全通信 |

---

## ✨ 主な機能

### 1. AST解析による高精度な分割

```python
# 入力コード例
def calculate_metrics(data: List[Dict]) -> Dict[str, float]:
    """データからメトリクスを計算する"""
    total = sum(item['value'] for item in data)
    average = total / len(data)
    return {'total': total, 'average': average}
```

**抽出される情報**:
- 関数名: `calculate_metrics`
- 引数: `data: List[Dict]`
- 戻り値: `Dict[str, float]`
- Docstring: `"データからメトリクスを計算する"`
- 依存関係: `List`, `Dict`
- メトリクス: LOC、複雑度等

### 2. 多言語対応

| 言語 | 対応機能 | 抽出情報 |
|------|---------|---------|
| **Python** | ✅ 完全対応 | 関数、メソッド、クラス、デコレータ、型ヒント |
| **Rust** | ✅ 完全対応 | 関数、impl、トレイト、ライフタイム |
| **Go** | ✅ 完全対応 | 関数、メソッド、構造体、インターフェース |
| **Java** | ✅ 完全対応 | メソッド、クラス、アノテーション、継承 |
| **C** | ✅ 完全対応 | 関数、構造体、マクロ |
| **C++** | ✅ 完全対応 | 関数、クラス、テンプレート、名前空間 |

### 3. 柔軟な除外設定

`.ragignore`ファイルで除外パターンを指定（.gitignore互換）:

```
# テストコード
*_test.py
*_test.go
*Test.java

# 自動生成コード
generated/
*_pb2.py
*.proto.go

# ビルド成果物
build/
dist/
target/
```

### 4. 埋め込みモデルの柔軟な切り替え

gRPCリクエストにモデル名を指定可能:

```yaml
# デフォルトモデル
model_name: "jinaai/jina-embeddings-v2-base-code"

# 他のモデルに切り替え（サーバー側設定のみ）
# - microsoft/unixcoder-base
# - bigcode/starencoder
```

サーバー側の設定変更のみで、クライアント（Indexer）の変更は不要。

### 5. リポジトリごとのコレクション管理

```yaml
# プロジェクトAのインデックス化
collection_name: "project-a"

# プロジェクトBのインデックス化
collection_name: "project-b"
```

複数のリポジトリを独立して管理可能。

---

## 🎓 ユースケース

### 1. コード理解支援

**シナリオ**: 新しいプロジェクトに参加したエンジニアが、特定の機能の実装箇所を探したい

```
質問: "ユーザー認証を処理している関数はどこ？"
↓ RAG検索
結果: auth/handlers.py の authenticate_user() 関数
      auth/middleware.py の verify_token() 関数
      （関連コードを含めてLLMに提供）
```

### 2. リファクタリング支援

**シナリオ**: 特定の関数を変更する前に、影響範囲を調査したい

```
質問: "calculate_discount() 関数を使っている箇所は？"
↓ メタデータ検索（calls フィールド）
結果: 5つのファイルで呼び出し元を特定
      （影響範囲を可視化）
```

### 3. バグ調査

**シナリオ**: エラーログから関連するコードを探したい

```
エラー: "NullPointerException in processPayment"
↓ RAG検索
結果: payment/processor.java の processPayment() メソッド
      payment/validator.java の validatePayment() メソッド
      （バグの原因箇所を特定）
```

### 4. ドキュメント生成

**シナリオ**: APIドキュメントを自動生成したい

```
質問: "REST API エンドポイントの一覧を作成して"
↓ メタデータ検索（decorators フィールドで @app.route を検索）
結果: 全エンドポイント関数を抽出
      （ドキュメント自動生成の素材）
```

---

## 🚀 今後の展開

このツールは、ローカルLLM環境でのコード検索・理解を支援する基盤として開発されています。

**関連プロジェクト（予定）**:
- **MCP Server**: Code RAG Indexerで登録されたデータを検索し、LLMに提供
- **CLI Chat Interface**: ターミナルからコードについて質問できるインターフェース
- **VS Code Extension**: エディタ統合による開発フロー支援

---

## 📋 システム要件

### Indexer（Goバイナリ）

- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 11+
- **メモリ**: 最小4GB、推奨8GB
- **ディスク**: ソースコードサイズに依存（通常数GB程度）

### 埋め込みサーバー（Docker）

- **Docker**: 20.10+
- **メモリ**: 最小8GB、推奨16GB（モデルロード用）
- **GPU**: オプション（CUDA対応GPUがあれば高速化）
- **ディスク**: 10GB以上（モデルキャッシュ用）

### Qdrant（Docker）

- **Docker**: 20.10+
- **メモリ**: 最小2GB、推奨4GB
- **ディスク**: ベクトルデータサイズに依存（100万関数で約5GB）

---

## 📚 ドキュメント

- [要件定義書](docs/requirements.md)
- [基本設計書](docs/design.md)
- [API仕様書](docs/api.md)（作成予定）
- [インストールガイド](docs/installation.md)（作成予定）
- [使用方法](docs/usage.md)（作成予定）

---

## 🗺️ ロードマップ

### Phase 1: 基本機能実装（現在）
- [x] 要件定義
- [x] 基本設計
- [ ] Python埋め込みサーバー実装
- [ ] Go Indexer実装
- [ ] E2Eテスト

### Phase 2: 機能拡張
- [ ] Gitリポジトリクローン機能
- [ ] 差分更新機能（変更ファイルのみ再登録）
- [ ] 周辺コンテキストのベクトル化
- [ ] Web UI（進捗確認）

### Phase 3: エコシステム構築
- [ ] MCP Server実装
- [ ] CLI Chat Interface実装
- [ ] VS Code Extension実装
- [ ] パフォーマンス最適化

---

## 🤝 コントリビューション

本プロジェクトは現在開発中です。コントリビューションガイドラインは実装完了後に公開予定です。

---

## 📄 ライセンス

MIT License

---

## 🙏 謝辞

本プロジェクトは以下のオープンソースプロジェクトを活用しています：

- [Tree-sitter](https://tree-sitter.github.io/) - 高速なパーサージェネレータ
- [Qdrant](https://qdrant.tech/) - ベクトルデータベース
- [Jina AI](https://jina.ai/) - コード特化型埋め込みモデル
- [gRPC](https://grpc.io/) - 高性能RPC フレームワーク

---

## 📧 お問い合わせ

- **Issues**: [GitHub Issues](https://github.com/yourorg/code-rag-indexer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourorg/code-rag-indexer/discussions)

---

**Built with ❤️ for developers who love clean code and powerful search**