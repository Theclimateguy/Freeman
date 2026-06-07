# Freeman — Production Roadmap

> Версия документа: 2026-06-07 (обновлён по итогам Тапов 1–4)  
> Базовый релиз: **v3.3.1** (локальная копия синхронизирована с `origin/main`, расхождений нет)

---

## 0. Статус релиза

| Параметр | Значение |
|---|---|
| Локальный HEAD | `2044696` — Release 3.3.1 |
| Remote `origin/main` HEAD | `2044696` — идентичен |
| Последний GitHub Release | `v3.3.1`, опубликован 2026-06-01 |
| Артефакты | `freeman-3.3.1-py3-none-any.whl`, `freeman_connectors-3.3.1-py3-none-any.whl` |
| Дополнительная ветка | `origin/freeman_lite` — stripped-down runtime (1.0.0), не влияет на main |

**Вывод:** локальная копия точно соответствует опубликованному релизу. Дополнительных изменений на удалённой стороне нет.

---

## 1. Диагностика: что мешает production

Ниже — конкретные проблемы, ранжированные по приоритету. Это не «хотелки»,
а вещи, которые реально сломаются под нагрузкой или в CI агент-кодера.

### P0 — Блокеры

| # | Проблема | Где | Симптом в production |
|---|---|---|---|
| P0-1 | `stream_runtime.py` — 2600 строк, один файл делает всё | `freeman/runtime/stream_runtime.py` | Невозможно протестировать отдельные фазы, любая правка ломает соседние ветки логики |
| P0-2 | KnowledgeGraph хранится целиком в JSON, загружается в RAM при старте | `freeman/memory/knowledgegraph.py` | При ~10k узлов — медленный старт, отсутствие concurrent writes, нет транзакций |
| P0-3 | Автономный ontology repair мутирует live-граф без транзакционной защиты | `freeman/agent/consciousness.py`, `stream_runtime.py` | При сбое в середине repair — частично применённая схема, неконсистентный граф |
| P0-4 | Нет health-check / readiness endpoint | отсутствует | Оркестратор (k8s, supervisord) не может определить, жив ли daemon |
| P0-5 | Секреты (API keys) в переменных окружения без валидации | `freeman/llm/*.py` | Агент стартует, делает несколько LLM-вызовов и только потом падает с auth error |

### P1 — Серьёзные

| # | Проблема | Где |
|---|---|---|
| P1-1 | `realworld/test_a_experiment.py` и `test_a_preflight.py` — production-код в пакете с именем `test_*` | `freeman/realworld/` |
| P1-2 | `verifier/fixed_point.py` — compatibility wrapper, но ссылки в коде смешаны; второй файл — orphan риск | `freeman/verifier/` |
| P1-3 | Нет rate-limiting / backpressure на signal ingestion queue | `freeman/agent/signalingestion.py` |
| P1-4 | Нет structured logging (только `logging.getLogger`), нет correlation ID | все runtime-файлы |
| P1-5 | Budget ledger пишется в JSONL без fsync-гарантий | `freeman/agent/costmodel.py` |
| P1-6 | Нет Dockerfile / compose для воспроизводимого запуска | отсутствует |

### P2 — Важные для надёжности

| # | Проблема |
|---|---|
| P2-1 | Нет circuit breaker для LLM-провайдеров (обрыв → бесконечные retries) |
| P2-2 | `hive_runtime` не имеет graceful shutdown при SIGTERM |
| P2-3 | Нет метрик (Prometheus / OpenTelemetry) — невозможно строить алерты |
| P2-4 | Forecast verification не покрыта тестом на clock skew (world.t vs runtime_step) |
| P2-5 | Нет документированного upgrade path при изменении схемы KG между версиями |

---

## 2. Архитектура целевого состояния

```
┌─────────────────────────────────────────────────────────┐
│                      Freeman Daemon                      │
│                                                          │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────┐  │
│  │  Signal  │──▶│  Ingestion    │──▶│  Analysis      │  │
│  │  Sources │   │  Engine       │   │  Pipeline      │  │
│  └──────────┘   └───────────────┘   └────────────────┘  │
│        ▲               │                    │            │
│        │        ┌──────▼──────┐     ┌───────▼──────┐   │
│   Connectors    │ Pending     │     │ World State  │   │
│   (external)    │ Queue       │     │ (versioned)  │   │
│                 └─────────────┘     └──────────────┘   │
│                                            │            │
│                          ┌─────────────────▼──────────┐ │
│                          │   KnowledgeGraph            │ │
│                          │   (pluggable backend)       │ │
│                          │   JSON(dev) / SQLite(prod)  │ │
│                          └────────────────────────────┘ │
│                                                          │
│  ┌────────────────────┐   ┌──────────────────────────┐  │
│  │  ConsciousnessEng  │   │  HiveRuntime             │  │
│  │  (read-only KG)    │   │  ingestor/planner/...    │  │
│  └────────────────────┘   └──────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Interface Layer                                  │    │
│  │  CLI · MCP Server · REST API · Health endpoint   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Observability                                    │    │
│  │  Structured logs (JSON) · Prometheus metrics      │    │
│  │  Budget ledger · Ontology audit trail             │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. План рефакторинга по тапам

### Тап 1 — Structural cleanup (≈ 3-5 дней, без изменения поведения)

**Цель:** убрать structural smell без изменения внешнего поведения. Все тесты должны проходить до и после.

#### 1.1 Распутать `stream_runtime.py`

Разбить на четыре модуля с чёткими интерфейсами:

```
freeman/runtime/
  stream_runtime.py          ← точка входа (argparse + main loop, <150 строк)
  bootstrap.py               ← bootstrap_mode dispatch + ETL фазы
  lifecycle.py               ← checkpoint resume/save, graceful shutdown
  signal_loop.py             ← poll loop, pending queue, signal processing
  query_handlers.py          ← --query forecasts/anomalies/semantic/answer
```

**Гардрейл:** `stream_runtime.py` после рефакторинга должен быть ≤ 200 строк.  
**Тест:** `pytest tests/test_runtime.py` проходит без изменений.

#### 1.2 Удалить `verifier/fixed_point.py`

`fixed_point.py` — compatibility wrapper, все импорты переключить на `fixedpoint.py` напрямую, затем удалить обёртку.

```bash
grep -r "fixed_point" freeman/ tests/ --include="*.py" -l
# исправить все -> fixedpoint
# удалить freeman/verifier/fixed_point.py
```

**Гардрейл:** `grep -r "from freeman.verifier.fixed_point" .` — 0 совпадений.

#### 1.3 Переименовать `realworld/test_a_*` и `test_c_*`

Это не тесты, это production experiment runners. Переименовать:

```
test_a_experiment.py   →  manifold_experiment.py
test_a_preflight.py    →  manifold_preflight.py
test_c_cross_domain.py →  cross_domain_runner.py
```

Обновить `tests/test_test_a_experiment.py` и смежные.

#### 1.4 Добавить `__all__` к публичным пакетам

`freeman/agent/__init__.py`, `freeman/memory/__init__.py`, `freeman/runtime/__init__.py` — явно экспортировать публичный API.

---

### Тап 2 — Reliability & Observability (≈ 1 неделя)

#### 2.1 Structured logging + Correlation ID

Заменить `logging.getLogger("stream_runtime")` на единый логгер с JSON-форматом:

```python
# freeman/logging_config.py
import logging, json, uuid

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        })

def get_logger(name: str, run_id: str | None = None) -> logging.Logger:
    ...
```

Каждый runtime-цикл получает `run_id = uuid4()`, который прокидывается через все вызовы.

**Тест:** `test_logging.py` — проверить, что log output валидный JSON.

#### 2.2 Health endpoint

Добавить `freeman/runtime/health.py`:

```python
class HealthState:
    last_signal_at: datetime | None
    last_kg_write_at: datetime | None
    world_t: int
    budget_remaining_usd: float
    status: Literal["ok", "degraded", "error"]

def get_health(runtime_state) -> HealthState: ...
```

Выставить через CLI: `freeman health --config ...` (JSON output).  
Выставить через MCP: добавить tool `freeman_health`.

**Тест:** `test_health.py` — проверить все статусы.

#### 2.3 KG write защита при ontology repair

Обернуть ontology repair в атомарный паттерн:

```python
# Перед repair — сохранить snapshot KG
# После успешного repair — commit
# При исключении — откатить к snapshot
with kg.transaction():
    apply_ontology_repair(...)
```

Реализовать `KnowledgeGraph.transaction()` как context manager с rollback через deepcopy (для JSON backend) или savepoint (для SQLite backend в будущем).

**Тест:** `test_repair_atomicity.py` — искусственно прервать repair на середине, проверить, что KG не изменился.

#### 2.4 Budget ledger fsync

```python
# costmodel.py — при записи ledger
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
    f.flush()
    os.fsync(f.fileno())
```

#### 2.5 LLM circuit breaker

Добавить `freeman/llm/circuit_breaker.py`:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60): ...
    def call(self, fn, *args, **kwargs): ...
    # state: CLOSED / OPEN / HALF_OPEN
```

Обернуть все `client.chat(...)` вызовы в `llm/adapter.py`.

**Тест:** `test_circuit_breaker.py` — проверить переход CLOSED→OPEN→HALF_OPEN.

#### 2.6 Graceful shutdown для hive_runtime

```python
# hive_runtime.py
import signal

_shutdown = threading.Event()

def _handle_sigterm(sig, frame):
    LOGGER.info("SIGTERM received, finishing current cycle")
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_sigterm)

# В main loop:
while not _shutdown.is_set():
    run_hive_cycle(...)
```

**Тест:** `test_hive_shutdown.py` — послать SIGTERM, проверить чистое завершение.

---

### Тап 3 — Storage backend (≈ 1 неделя)

**Цель:** сделать KnowledgeGraph backend подключаемым, добавить SQLite как production-вариант.

#### 3.1 Ввести `KGBackend` протокол

```python
# freeman/memory/backends/base.py
from typing import Protocol

class KGBackend(Protocol):
    def load(self) -> dict: ...
    def save(self, data: dict) -> None: ...
    def transaction(self) -> ContextManager: ...
    def node_count(self) -> int: ...
```

#### 3.2 `JsonKGBackend` — рефакторинг существующего

Текущий JSON backend оборачивается в `JsonKGBackend(path)` без изменения логики.

#### 3.3 `SqliteKGBackend` — новый

```python
# freeman/memory/backends/sqlite.py
# Схема: nodes(id, type, data JSON), edges(source, target, relation, data JSON)
# Индексы: nodes(type), nodes(status), edges(source), edges(target)
# transaction() → sqlite3 SAVEPOINT
```

Конфигурация через `config.yaml`:

```yaml
memory:
  backend: sqlite          # json | sqlite
  sqlite_path: ./data/kg.db
```

**Тест:** `test_kg_backends.py` — параметризованные тесты против обоих backend через один интерфейс.

---

### Тап 4 — Containerization & Deployment (≈ 3 дня)

#### 4.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY freeman/ freeman/
COPY packages/ packages/

RUN pip install --no-cache-dir -e ".[semantic,geo,redis]" \
    && pip install --no-cache-dir -e ./packages/freeman-connectors

COPY config.yaml.example /app/config.yaml

ENV PYTHONUNBUFFERED=1
ENV FREEMAN_CONFIG=/app/config.yaml

HEALTHCHECK --interval=30s --timeout=5s \
  CMD freeman health --config $FREEMAN_CONFIG || exit 1

ENTRYPOINT ["python", "-m", "freeman.runtime.stream_runtime"]
```

#### 4.2 docker-compose.yml

```yaml
services:
  freeman:
    build: .
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "freeman", "health", "--config", "/app/config.yaml"]
      interval: 30s

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

#### 4.3 Секреты

Валидация API-ключей при старте:

```python
# freeman/runtime/startup_checks.py
def validate_config(config: dict) -> list[str]:
    """Returns list of critical errors before runtime starts."""
    errors = []
    provider = config.get("llm", {}).get("provider", "none")
    if provider not in ("none", "ollama"):
        key = os.environ.get("LLM_API_KEY") or config.get("llm", {}).get("api_key")
        if not key:
            errors.append(f"LLM provider '{provider}' requires LLM_API_KEY")
    return errors
```

Вызывать до старта основного loop.

---

### Тап 5 — Metrics (≈ 2 дня)

Добавить `freeman/runtime/metrics.py` с Prometheus-совместимым output:

```python
# Счётчики
signals_ingested_total
signals_filtered_total
ontology_repairs_total
llm_calls_total{provider, task_type}
llm_errors_total{provider, error_type}

# Gauge
world_t
kg_node_count
budget_spent_usd
budget_remaining_usd
active_forecasts

# Histogram
signal_processing_seconds
analysis_pipeline_seconds
llm_call_seconds{provider}
```

Endpoint: `freeman metrics --config ...` (текстовый Prometheus format).  
Опционально: `pushgateway_url` в конфиге для push-mode.

---

### Тап 6 — Test suite hardening (≈ 1 неделя)

#### 6.1 Матрица тестов для агент-кодера

Категории тестов и их предназначение:

```
tests/
  unit/                    # изолированные, без I/O
    test_verifier_*.py
    test_scorer.py
    test_consciousness_*.py
    test_costmodel.py
    test_circuit_breaker.py
  integration/             # работают с реальными файлами, без LLM
    test_runtime.py
    test_kg_backends.py
    test_signal_loop.py
    test_repair_atomicity.py
    test_health.py
  behavioral/              # end-to-end, с mock LLM
    test_bootstrap_strategies.py
    test_ontology_repair_cycle.py
    test_forecast_lifecycle.py
    test_hive_dispatch.py
  smoke/                   # CLI smoke tests (уже есть в CI)
    test_cli_query.py
    test_cli_ask.py
    test_cli_what_if.py
```

#### 6.2 Обязательные новые тесты

| Тест | Что проверяет |
|---|---|
| `test_repair_atomicity.py` | KG не мутирует при обрыве ontology repair |
| `test_world_t_monotonic.py` | `world.t` никогда не откатывается после fallback |
| `test_budget_fsync.py` | Запись в cost_ledger.jsonl выживает после kill -9 |
| `test_kg_backend_parity.py` | JSON и SQLite backend дают одинаковый результат на одних данных |
| `test_hive_sigterm.py` | Hive завершается чисто при SIGTERM |
| `test_startup_validation.py` | Старт отказывает при отсутствующем API key |
| `test_health_states.py` | Health endpoint возвращает `degraded` при просроченном signal |

#### 6.3 Минимальный coverage gate

В `pytest.ini`:

```ini
[tool.pytest.ini_options]
addopts = "--cov=freeman --cov-fail-under=75"
```

Текущий hive-mind coverage gate — 80%. Распространить на весь пакет.

#### 6.4 Mutation testing (опционально, рекомендуется для core/)

```bash
pip install mutmut
mutmut run --paths-to-mutate freeman/core/ freeman/verifier/
mutmut results
```

Целевой mutation score для `core/` и `verifier/` — ≥ 70%.

---

### Тап 7 — Upgrade path & versioning (≈ 2 дня)

#### 7.1 KG schema migration

Добавить `version` поле в корень `kg_state.json`:

```json
{"__freeman_kg_version": 1, "nodes": [...], "edges": [...]}
```

`KnowledgeGraph.load()` проверяет версию и запускает migrators:

```python
MIGRATIONS = {
    0: migrate_v0_to_v1,
    1: migrate_v1_to_v2,
}
```

#### 7.2 CHANGELOG policy

Обязательная запись в `## Unreleased` при каждом PR, который меняет:
- публичный API
- формат KG / checkpoint
- конфигурационные ключи

Проверяется в CI: `scripts/check_changelog.py` — если `## Unreleased` содержит `No unreleased changes`, а diff затрагивает `freeman/`, — fail.

---

## 4. Гардрейлы для production-агента

### Инварианты, которые нельзя нарушать

```
[I-1] world.t только растёт — никогда не уменьшается
[I-2] KG мутирует только через явные методы update_node/add_node/add_relation
[I-3] LLM не пишет в WorldState напрямую — только через ParameterEstimator
[I-4] Ontology repair всегда атомарен (откат при исключении)
[I-5] Budget hard limit останавливает выполнение до исчерпания бюджета
[I-6] Старт отказывает при невалидном конфиге (fail-fast)
```

### Операционные ограничения

```yaml
# Рекомендуемые пределы в config.yaml для production
agent:
  budget_usd_per_day: 2.0
  cost_governance:
    signal_processing_max_usd: 0.01
    ontology_repair_max_usd: 0.05
    answer_generation_max_usd: 0.02

runtime:
  kg_snapshots:
    enabled: true
    max_snapshots: 48        # 24 часа при 30-мин интервале
  pending_queue_max_size: 500
  max_ontology_repairs_per_hour: 3

memory:
  reconciler:
    merge_threshold: 0.85
    compaction_interval: 100
```

### Алерты (минимальный набор)

| Условие | Severity |
|---|---|
| `budget_remaining_usd < 0.10` | warning |
| `ontology_repairs_total` растёт > 5/час | warning |
| `signals_ingested_total` не растёт > 30 минут | error |
| `health.status == "error"` | critical |
| `llm_errors_total` > 10 за 5 минут | error |
| `world_t` не меняется > 2 × `analysis_interval_seconds` | warning |

---

## 5. Первый production-тест: климатические новости

> Реализуется **после** завершения Тапов 1–4.

### Сценарий: Freeman Climate Monitor

**Цель:** верифицировать, что Freeman может автономно:
1. Принять климатический brief
2. Построить онтологию через `llm_synthesize`
3. Непрерывно потреблять RSS-ленты климатических новостей
4. Обнаруживать аномалии и обновлять прогнозы
5. Отвечать на семантические вопросы по накопленной памяти

### Входные данные

```
brief:    examples/domain_brief_climate_news.md
config:   config.climate.yaml
sources:  RSS (уже поддерживается через freeman-connectors)
duration: 24 часа (--hours 24)
```

### Acceptance criteria

| Критерий | Метрика |
|---|---|
| Bootstrap успешен | `bootstrap_package.json` содержит `materialization_path != null` |
| Онтология валидна | Verifier level0 + level1 pass без ошибок |
| Сигналы потребляются | `signals_ingested_total > 50` за 24 часа |
| Прогнозы создаются | `active_forecasts > 0` через 2 часа |
| Семантический ответ работает | `freeman ask "What are the main climate risks?" --config config.climate.yaml` возвращает `answer_generated: true` |
| Бюджет соблюдён | `spent_usd < budget_usd_per_day` |
| Нет crashed world.t | `world.t` монотонно растёт в event_log.jsonl |

### Автотест-обёртка

```python
# tests/integration/test_climate_24h.py
# Помечен @pytest.mark.slow — не запускается в обычном CI
# Запускается отдельно: pytest -m slow tests/integration/test_climate_24h.py

@pytest.mark.slow
def test_climate_monitor_acceptance(tmp_path):
    """24-hour climate news monitor acceptance test."""
    ...
```

---

## 6. Порядок выполнения для агент-кодера

```
Тап 1: Structural cleanup        — ✅ ЗАКРЫТ (303 passed)
Тап 2: Reliability               — ✅ ЗАКРЫТ (structured logging, health, KG tx,
                                              budget fsync, circuit breaker,
                                              hive SIGTERM, startup validation,
                                              ingestion backpressure)
Тап 3: Storage backend           — ✅ ЗАКРЫТ (SQLite backend, parity tests,
                                              config-driven selection)
Тап 4: Containerization          — ✅ ЗАКРЫТ (Dockerfile, compose, .dockerignore,
                                              FREEMAN_CONFIG env; docker build
                                              sha256:699fb4d3, ~267 MB,
                                              health/entrypoint/import verified)
Тап 5: Metrics                   — ✅ ЗАКРЫТ (`freeman metrics`,
                                              Prometheus text output,
                                              counters/gauges/histograms)
Тап 6: Test suite hardening      — coverage gate 75%, mutation tests для core/
Тап 7: Versioning                — KG schema migrations, CHANGELOG policy
Климатический тест               — после релиза 3.4.0 или параллельно
```

### Findings из верификации

| # | Файл | Статус |
|---|---|---|
| F-1 | `freeman/logging_config.py` | ✅ снят: `json_mode` поддержан как alias для `json_logs` |
| F-2 | `freeman/runtime/startup_checks.py` | ✅ снят: `agent.budget_usd_per_day` default = `0.5`, добавлен guardrail-тест |
| F-3 | `.dockerignore` | ✅ снят: глобальный `*.jsonl` заменён на runtime/data/logs scoped patterns |
| F-4 | `docker-compose.yml` | ✅ снят: Redis healthcheck + `depends_on.condition=service_healthy` |

### Правила для агент-кодера

1. **Каждый тап — отдельный PR.** Не смешивать structural cleanup с новой функциональностью.
2. **Тесты пишутся до или одновременно с кодом.** Не после.
3. **`pytest -q` должен проходить после каждого коммита.** Если нет — откатить.
4. **Инварианты I-1 — I-6 не нарушаются.** Если PR их нарушает — отклонить.
5. **`stream_runtime.py` не растёт.** Любое добавление функциональности — только через декомпозицию.
6. **LLM-зависимые тесты** помечаются `@pytest.mark.llm` и не запускаются в основном CI без явного флага.
7. **Формат KG** при изменении требует migration-скрипт и bump `__freeman_kg_version`.

---

## 7. Definition of Done для production-ready

Freeman считается production-ready когда:

- [x] `stream_runtime.py` ≤ 200 строк (точка входа)
- [ ] `pytest --cov=freeman --cov-fail-under=75` проходит в CI
- [x] Health endpoint отвечает за < 100ms
- [x] Graceful shutdown при SIGTERM (< 5 секунд)
- [x] Ontology repair атомарен (тест подтверждён)
- [x] Docker build работает из чистого clone
- [x] `world.t` monotonicity тест зелёный
- [x] Budget governance тест зелёный
- [x] `freeman status` показывает live budget из ledger
- [x] CHANGELOG содержит upgrade notes для каждого breaking change
- [ ] Климатический 24h тест пройден хотя бы один раз

---

*Документ создан на основе анализа `v3.3.1` и обновлён для релиза `v3.4.0`. При значительных изменениях архитектуры — обновить секции 1 и 2.*
