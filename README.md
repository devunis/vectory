# Vectory

여러 벡터 데이터베이스를 생성하고 관리할 수 있는 벡터 DB 플랫폼입니다.

## 주요 기능

- **다중 컬렉션 관리** — 독립적인 벡터 DB를 여러 개 생성/삭제/조회
- **유사도 검색** — cosine, euclidean, dot_product 거리 메트릭 지원
- **메타데이터 필터링** — 검색 시 메타데이터 조건으로 결과 필터링
- **영속화** — JSON 파일 기반 자동 저장/로드
- **REST API** — FastAPI 기반 HTTP API + Swagger 문서 자동 생성
- **CLI** — 터미널에서 바로 사용 가능한 명령어 인터페이스

## 설치

```bash
pip install -e .
```

개발 의존성 포함:

```bash
pip install -e ".[dev]"
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
