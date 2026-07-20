# Vectory

여러 벡터 데이터베이스를 생성하고 관리할 수 있는 벡터 DB 플랫폼입니다.

## 주요 기능

- **다중 컬렉션 관리** — 독립적인 벡터 DB를 여러 개 생성/삭제/조회
- **유사도 검색** — cosine, euclidean, dot_product 거리 메트릭 지원
- **메타데이터 필터링** — 검색 시 메타데이터 조건으로 결과 필터링
- **영속화** — JSON 파일 기반 자동 저장/로드
- **REST API** — FastAPI 기반 HTTP API + Swagger 문서 자동 생성
- **CLI** — 터미널에서 바로 사용 가능한 명령어 인터페이스
- **문서 파싱** — 선택 의존성으로 PaddleOCR 로컬 파싱과 MinerU API 파싱 지원
- **RAG 검색 전략** — BM25, 벡터 검색, Hybrid Search, RRF, MMR, HyDE-style, Adaptive/Corrective/Graph/RAPTOR 검색 지원

## 설치

```bash
pip install -e .
```

개발 의존성 포함:

```bash
pip install -e ".[dev]"
```

문서 파싱 기능 포함:

```bash
pip install -e ".[parse]"
```

## 빠른 시작

### Python 코드

```python
from vectory.engine import CollectionManager

mgr = CollectionManager.with_file_storage("./my_data")

# 컬렉션 생성
col = mgr.create_collection("docs", dimension=3, metric="cosine")

# 벡터 삽입
col.insert(
    vectors=[[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0]],
    ids=["a", "b", "c"],
    metadata=[{"label": "x"}, {"label": "y"}, {"label": "z"}],
)

# 유사도 검색
results = col.search([1, 0, 0], top_k=2)
for r in results:
    print(f"{r.id}  score={r.score:.4f}  {r.metadata}")

# 저장
mgr.save_collection("docs")
```

### REST API 서버

```bash
python -m vectory serve --port 8000
```

실행 후 `http://localhost:8000/docs`에서 Swagger UI를 확인할 수 있습니다.

#### API 예시

```bash
# 컬렉션 생성
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_db", "dimension": 3}'

# 벡터 삽입
curl -X POST http://localhost:8000/collections/my_db/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[1,0,0],[0,1,0]], "ids": ["a","b"]}'

# 검색
curl -X POST http://localhost:8000/collections/my_db/search \
  -H "Content-Type: application/json" \
  -d '{"query": [1,0,0], "top_k": 2}'
```

### CLI

```bash
python -m vectory create my_db --dimension 3 --metric cosine
python -m vectory list
python -m vectory info my_db
python -m vectory search my_db "[1.0, 0.0, 0.0]" --top-k 2
python -m vectory delete my_db
```

`--data-dir`는 전역 옵션이라 서브커맨드 앞에 위치해야 합니다.

```bash
python -m vectory --data-dir ./my_data create my_db --dimension 3
python -m vectory --data-dir ./my_data list
```

### 문서 파싱

PaddleOCR 로컬 OCR:

```bash
python -m vectory parse ./sample.png --provider paddleocr
```

PaddleOCR 문서 구조 파싱:

```bash
python -m vectory parse ./sample.png --provider paddleocr --mode structure --output-dir ./parsed
```

MinerU Agent 경량 API 파싱:

```bash
python -m vectory parse ./sample.pdf --provider mineru --fetch-markdown
python -m vectory parse https://example.com/sample.pdf --provider mineru --source-type url --fetch-markdown
```

REST API에서도 같은 기능을 사용할 수 있습니다.

```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"provider": "mineru", "source": "https://example.com/sample.pdf", "source_type": "url"}'
```

### RAG 검색

텍스트 파일을 chunk로 나눠 RAG 컬렉션에 넣습니다. 기본 임베딩은 외부 의존성 없는 해시 기반 baseline이라, 운영에서는 별도 embedding provider를 붙이는 용도로 확장하면 됩니다.

```bash
python -m vectory rag ingest docs ./sample.txt \
  --document-id sample \
  --chunk-size 200 \
  --chunk-overlap 40
```

검색 전략은 `vector`, `bm25`, `hybrid`를 지원합니다. `hybrid`는 벡터 검색과 BM25 결과를 Reciprocal Rank Fusion(RRF)으로 합칩니다.

```bash
python -m vectory rag search docs "에러 코드 TS-999 해결 방법" --strategy hybrid --top-k 5
```

고급 전략도 사용할 수 있습니다.

- `adaptive`: 쿼리 특성에 따라 BM25/Hybrid/MMR/Reranker를 휴리스틱으로 라우팅
- `corrective`: 검색 결과 품질을 평가하고 낮으면 BM25 보정 검색을 추가 수행
- `raptor`: ingest 시 만든 계층 summary chunk를 활용
- `graph`: entity-like token을 기준으로 연결 chunk를 확장

```bash
python -m vectory rag ingest docs ./sample.txt --enable-raptor
python -m vectory rag search docs "복잡한 관계를 비교해줘" --strategy adaptive
python -m vectory rag search docs "검색 결과가 부족할 수 있는 질문" --strategy corrective
python -m vectory rag search docs "문서 전체 주제" --strategy raptor
python -m vectory rag search docs "Alice ProjectX 관계" --strategy graph
```

초기 검색 결과를 다시 정렬하려면 lexical reranker를 적용할 수 있습니다.

```bash
python -m vectory rag search docs "정확한 키워드" --strategy hybrid --reranker lexical
```

쿼리 확장/RAG-Fusion 스타일 검색은 `--expand`를 여러 번 넘겨 사용할 수 있습니다.

```bash
python -m vectory rag search docs "문서 파싱" \
  --expand "PDF 테이블 추출" \
  --expand "OCR 스캔 문서 인식"
```

HyDE-style 검색은 외부 LLM이 만든 hypothetical document를 `--hyde`로 넘겨 벡터 검색 질의로 사용합니다.

```bash
python -m vectory rag search docs "영수증 텍스트 추출" \
  --strategy vector \
  --hyde "PaddleOCR로 스캔된 영수증에서 품목과 금액을 추출하는 문서"
```

REST API:

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs", "text": "Vectory supports hybrid RAG search.", "document_id": "doc1"}'

curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs", "query": "hybrid RAG", "strategy": "hybrid"}'
```

## 프로젝트 구조

```
vectory/
├── engine/
│   ├── distance.py    # 거리 메트릭 (cosine, euclidean, dot_product)
│   ├── collection.py  # 단일 벡터 컬렉션 (CRUD + 검색)
│   └── manager.py     # 다중 컬렉션 관리
├── storage/
│   └── backend.py     # 스토리지 백엔드 (파일 / 인메모리)
├── api/
│   ├── schemas.py     # Pydantic 요청/응답 스키마
│   └── server.py      # FastAPI REST API
├── parsing/           # PaddleOCR / MinerU 문서 파싱 어댑터
├── rag/               # Chunking, BM25, Hybrid/RRF/MMR, Adaptive/Corrective/Graph/RAPTOR 검색
└── cli/
    └── main.py        # Click CLI
```

## 지원 거리 메트릭

| 메트릭 | 설명 | 값이 작을수록 |
|---|---|---|
| `cosine` | 코사인 거리 (1 - 유사도) | 유사함 |
| `euclidean` | 유클리드 거리 | 유사함 |
| `dot_product` | 내적의 음수 | 유사함 |

## 테스트

```bash
python -m pytest tests/ -v
```

## 라이선스

MIT
