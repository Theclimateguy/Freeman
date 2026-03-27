# Freeman: Техническое задание на разработку универсального симулятора
**Версия:** 1.0  
**Репозиторий:** github.com/Theclimateguy/Freeman  
**Связанный проект:** github.com/Theclimateguy/GIM (GIM15 — референсная реализация геополитического домена)  

---

## 1. Контекст и цель

GIM15 — зрелый симулятор геополитики, экономики и климата с доказанной архитектурой:
`sim_bridge.py` → `game_runner.py` → `crisis_metrics.py` → `briefing.py`. Он работает,
но намертво привязан к одному домену: 57 стран, фиксированные типы кризисов, конкретные
ресурсы и исходы закодированы как константы.

**Freeman** — универсальное ядро симулятора, из которого GIM15 становится одним из доменных
профилей. Любой другой домен (рынок воды, цепочки поставок, социальные сети, здравоохранение,
энергосистема) подключается через единую JSON-схему без изменения кода ядра.

**Конечная цель Freeman:** стать инструментом (tool) для LLM-агента Freeman-Agent, который
компилирует домен из текстового описания, запускает симуляцию, верифицирует результат и
интерпретирует траектории. Но сначала — правильно работающий симулятор.

---

## 2. Принципы проектирования

1. **Domain-agnostic core.** Ядро не знает что такое "страна", "GDP" или "военный конфликт".
   Оно знает акторов, ресурсы, исходы и операторы перехода.

2. **GIM15-совместимость.** GIM15 должен запускаться поверх Freeman как доменный профиль
   без потери функциональности. Это критерий правильности реализации.

3. **Верификация встроена, не опциональна.** Каждый шаг симуляции проходит через
   трёхуровневый верификатор. Нарушение уровня 0 — жёсткая остановка.

4. **Pluggable physics.** Форма оператора перехода выбирается per-resource из реестра.
   Ядро не диктует физику — домен её задаёт.

5. **Сериализуемость.** Любое состояние мира, домен и результат симуляции должны
   полностью сериализоваться в JSON. Это обязательное условие для LLM-интеграции.

6. **Детерминизм.** При одинаковом `seed` и одинаковых входах симулятор даёт одинаковый
   результат. Без этого невозможна верификация и воспроизводимость.

---

## 3. Архитектура Freeman

```
freeman/
├── core/
│   ├── types.py              # базовые типы: Actor, Resource, Relation, Outcome, Policy
│   ├── world.py              # WorldState — снапшот мира в момент t
│   ├── evolution.py          # Function Registry + все операторы перехода
│   ├── scorer.py             # softmax outcome scoring
│   ├── transition.py         # главный оператор T(S, pi, theta_D)
│   └── multiworld.py         # мультидоменная композиция через SharedResourceBus
├── verifier/
│   ├── level0.py             # математические инварианты (без домена)
│   ├── level1.py             # структурные тесты (без домена)
│   ├── level2.py             # sign-consistency (требует CausalDAG)
│   ├── fixed_point.py        # итерационная сходимость для циклических DAG
│   └── report.py             # VerificationReport: агрегация нарушений
├── domain/
│   ├── schema.py             # DomainSchema — JSON-контракт для описания домена
│   ├── compiler.py           # DomainCompiler: JSON → инициализированный WorldState
│   ├── registry.py           # DomainRegistry: каталог известных доменных профилей
│   └── profiles/
│       └── gim15.json        # доменный профиль GIM15 (референс)
├── game/
│   ├── runner.py             # GameRunner: итерация шагов + rolling verifier
│   ├── equilibrium.py        # Nash / CCE solver (перенос из GIM15/game_theory/)
│   └── result.py             # SimResult: траектории, исходы, confidence, violations
├── api/
│   ├── tool_api.py           # OpenAI function-calling совместимый API
│   └── mcp_server.py         # MCP server для LLM-интеграции
├── tests/
│   ├── test_level0.py
│   ├── test_level1.py
│   ├── test_level2.py
│   ├── test_fixed_point.py
│   ├── test_evolution.py
│   ├── test_compiler.py
│   ├── test_multiworld.py
│   └── test_gim15_compat.py  # GIM15 как доменный профиль — регрессионный тест
└── config.yaml
```

---

## 4. Модуль core/types.py

Базовые типы. Никакой бизнес-логики — только структуры данных.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any
import numpy as np

@dataclass
class Actor:
    id: str
    name: str
    state: Dict[str, float]        # произвольный вектор состояния
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Resource:
    id: str
    name: str
    value: float
    unit: str
    owner_id: Optional[str] = None  # None = глобальный ресурс
    min_value: float = 0.0
    max_value: float = float("inf")
    evolution_type: str = "stock_flow"  # ключ в EvolutionRegistry

@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str              # "trade", "alliance", "conflict", "dependency", ...
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class Outcome:
    id: str
    label: str
    scoring_weights: Dict[str, float]  # resource_id / state_key → weight
    description: str = ""

@dataclass
class CausalEdge:
    source: str                    # resource_id или state_key
    target: str
    expected_sign: Literal["+", "-"]
    strength: Literal["strong", "weak"] = "strong"

@dataclass
class Policy:
    actor_id: str
    actions: Dict[str, float]      # action_id → intensity [-1, 1]

@dataclass
class Violation:
    level: int                     # 0, 1, или 2
    check_name: str
    description: str
    severity: Literal["hard", "soft"]
    details: Dict[str, Any] = field(default_factory=dict)
```

---

## 5. Модуль core/world.py

```python
@dataclass
class WorldState:
    domain_id: str
    t: int                              # текущий временной шаг
    actors: Dict[str, Actor]
    resources: Dict[str, Resource]
    relations: List[Relation]
    outcomes: Dict[str, Outcome]
    causal_dag: List[CausalEdge]
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict:
        """Полная сериализация в JSON-совместимый dict."""
        ...

    @classmethod
    def from_snapshot(cls, data: Dict) -> "WorldState":
        """Восстановление из JSON."""
        ...

    def clone(self) -> "WorldState":
        """Глубокая копия для контрфактических симуляций."""
        ...
```

**Требования:**
- `snapshot()` / `from_snapshot()` должны быть обратно совместимы (round-trip без потерь)
- `clone()` создаёт независимую копию — изменения клона не затрагивают оригинал
- Все float-значения хранятся с точностью float64

---

## 6. Модуль core/evolution.py — Function Registry

Каждый тип оператора перехода реализует единый интерфейс:

```python
from abc import ABC, abstractmethod

class EvolutionOperator(ABC):
    @abstractmethod
    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0
    ) -> float:
        """Возвращает новое значение ресурса после одного шага."""
        ...

    @abstractmethod
    def stability_bound(self) -> float:
        """Верхняя граница роста за один шаг. Используется level1 verifier."""
        ...
```

### 6.1 Реализации

**LinearTransition** — гарантированно устойчивый:
```
R(t+1) = a * R(t) + b * sum(policy.actions) + c
Параметры: a ∈ (0, 1), b, c — из DomainSchema
```

**StockFlowTransition** — default, аналог GIM15:
```
R(t+1) = R(t) + phi(world, policy) - delta * R(t)
phi — настраиваемая функция притока (линейная по умолчанию)
delta ∈ (0, 1) — коэффициент убыли
Гарантия: R(t) >= 0 при phi >= 0 и delta ∈ (0,1)
```

**LogisticGrowthTransition** — для популяций, рынков:
```
R(t+1) = R(t) + r * R(t) * (1 - R(t)/K) + external(policy)
r — скорость роста, K — ёмкость
```

**ThresholdTransition** — для кризисов, переломных точек:
```
R(t+1) = f_low(R(t), policy)  если R(t) < theta
         f_high(R(t), policy) если R(t) >= theta
```

**CoupledTransition** — композиция нескольких операторов:
```
R(t+1) = sum(w_i * operator_i.step(...))
```

### 6.2 Реестр

```python
EVOLUTION_REGISTRY: Dict[str, Type[EvolutionOperator]] = {
    "linear":     LinearTransition,
    "stock_flow": StockFlowTransition,
    "logistic":   LogisticGrowthTransition,
    "threshold":  ThresholdTransition,
    "coupled":    CoupledTransition,
}

def get_operator(evolution_type: str, params: Dict) -> EvolutionOperator:
    cls = EVOLUTION_REGISTRY[evolution_type]
    return cls(**params)
```

---

## 7. Модуль core/transition.py — главный оператор

```python
def step_world(
    world: WorldState,
    policies: List[Policy],
    dt: float = 1.0
) -> tuple[WorldState, List[Violation]]:
    """
    Один шаг симуляции. Возвращает новое состояние и список нарушений.
    При нарушении уровня 0 (severity=hard) поднимает HardStopException.
    """
    next_world = world.clone()
    violations = []

    # 1. обновить каждый ресурс через его оператор
    policy_map = {p.actor_id: p for p in policies}
    for res_id, resource in world.resources.items():
        operator = get_operator(resource.evolution_type, resource.params)
        actor_policy = policy_map.get(resource.owner_id)
        new_value = operator.step(resource, world, actor_policy, dt)
        next_world.resources[res_id].value = new_value

    # 2. обновить состояния акторов (domain-specific, через actor_update_fn)
    for actor_id, actor in world.actors.items():
        next_world.actors[actor_id] = world.domain_actor_update(actor, world, policy_map)

    # 3. запустить rolling verifier
    v0 = level0_check(world, next_world)
    violations.extend(v0)
    if any(v.severity == "hard" for v in v0):
        raise HardStopException(violations=v0)

    next_world.t = world.t + 1
    return next_world, violations
```

---

## 8. Модуль verifier/ — трёхуровневая верификация

### 8.1 Level 0 — математические инварианты

Запускается **на каждом шаге**. Нарушение = жёсткая остановка.

```python
def level0_check(prev: WorldState, next: WorldState) -> List[Violation]:
    violations = []

    # conservation: sum ресурсов не растёт без внешних притоков
    total_prev = sum(r.value for r in prev.resources.values())
    total_next = sum(r.value for r in next.resources.values())
    external = next.metadata.get("exogenous_inflow", 0.0)
    if total_next > total_prev + external + EPSILON:
        violations.append(Violation(
            level=0, check_name="conservation",
            description=f"Resource sum grew {total_next - total_prev:.4f} without exogenous inflow",
            severity="hard"
        ))

    # nonnegativity
    for res_id, res in next.resources.items():
        if res.value < res.min_value - EPSILON:
            violations.append(Violation(
                level=0, check_name="nonnegativity",
                description=f"Resource {res_id} = {res.value:.4f} < min {res.min_value}",
                severity="hard"
            ))

    # probability simplex
    outcome_probs = score_outcomes(next)
    if abs(sum(outcome_probs.values()) - 1.0) > EPSILON:
        violations.append(Violation(
            level=0, check_name="probability_simplex",
            description=f"Outcome probs sum = {sum(outcome_probs.values()):.6f} != 1.0",
            severity="hard"
        ))

    # bounds
    for res_id, res in next.resources.items():
        if res.value > res.max_value + EPSILON:
            violations.append(Violation(
                level=0, check_name="bounds",
                description=f"Resource {res_id} = {res.value:.4f} > max {res.max_value}",
                severity="soft"
            ))

    return violations
```

### 8.2 Level 1 — структурные тесты

Запускается **один раз при инициализации домена** и при изменении параметров.

```python
def level1_check(world: WorldState, config: SimConfig) -> List[Violation]:
    violations = []

    # null_action_convergence: нулевые действия → стационарность за K шагов
    null_world = world.clone()
    null_policies = [Policy(a, {}) for a in world.actors]
    prev_state = null_world.snapshot()
    for k in range(config.convergence_check_steps):
        null_world, _ = step_world(null_world, null_policies)
        curr_state = null_world.snapshot()
        if state_distance(prev_state, curr_state) < config.convergence_epsilon:
            break
        prev_state = curr_state
    else:
        violations.append(Violation(
            level=1, check_name="null_action_convergence",
            description=f"World did not converge in {config.convergence_check_steps} steps under null policy",
            severity="soft"
        ))

    # spectral_radius: rho(J) < 1 для гарантии сходимости fixed-point
    J = compute_jacobian(world)
    rho = spectral_radius(J)
    if rho >= 1.0:
        violations.append(Violation(
            level=1, check_name="spectral_radius",
            description=f"Jacobian spectral radius = {rho:.4f} >= 1.0, fixed-point may not converge",
            severity="soft",
            details={"spectral_radius": rho}
        ))

    # shock_decay: единичный шок затухает за K шагов
    for res_id in list(world.resources.keys())[:3]:  # проверяем первые 3 ресурса
        decay_ok = check_shock_decay(world, res_id, config)
        if not decay_ok:
            violations.append(Violation(
                level=1, check_name="shock_decay",
                description=f"Shock on resource {res_id} does not decay",
                severity="soft"
            ))

    return violations
```

### 8.3 Level 2 — sign-consistency

Запускается **после каждого N шагов** (rolling) и при инициализации.

```python
def level2_check(
    world: WorldState,
    causal_dag: List[CausalEdge],
    delta: float = 0.01
) -> List[Violation]:
    violations = []
    for edge in causal_dag:
        # применяем малый шок к source
        shocked = world.clone()
        shocked = apply_delta(shocked, edge.source, delta)
        next_shocked, _ = step_world(shocked, [])
        next_base, _ = step_world(world, [])

        observed_delta = get_value(next_shocked, edge.target) - get_value(next_base, edge.target)
        observed_sign = "+" if observed_delta > 0 else "-"

        if observed_sign != edge.expected_sign and abs(observed_delta) > SIGN_EPSILON:
            violations.append(Violation(
                level=2, check_name="sign_consistency",
                description=f"Edge {edge.source} → {edge.target}: expected {edge.expected_sign}, got {observed_sign}",
                severity="soft" if edge.strength == "weak" else "hard",
                details={"edge": edge, "observed_delta": observed_delta}
            ))
    return violations
```

### 8.4 fixed_point.py — сходимость при циклических DAG

```python
def find_fixed_point(
    world: WorldState,
    causal_dag: List[CausalEdge],
    max_iter: int = 20,
    alpha: float = 0.1,
    tolerance: float = 1e-6
) -> tuple[WorldState, bool, int]:
    """
    Итерирует коррекции параметров до устранения sign-violations.
    Возвращает: (откорректированный world, сошлось, число итераций).
    """
    current = world.clone()
    for i in range(max_iter):
        violations = level2_check(current, causal_dag)
        sign_violations = [v for v in violations if v.check_name == "sign_consistency"]
        if not sign_violations:
            return current, True, i

        # применяем коррекцию по нарушениям
        corrections = compute_corrections(sign_violations, alpha)
        current = apply_corrections(current, corrections)

    return current, False, max_iter
```

---

## 9. Модуль core/scorer.py

```python
def score_outcomes(world: WorldState) -> Dict[str, float]:
    """
    Возвращает вероятностное распределение по исходам.
    p(o) = softmax(W * S)
    """
    raw_scores = {}
    for outcome_id, outcome in world.outcomes.items():
        score = 0.0
        for key, weight in outcome.scoring_weights.items():
            value = get_world_value(world, key)  # resource или actor state
            score += weight * value
        raw_scores[outcome_id] = score

    # softmax
    max_score = max(raw_scores.values())
    exp_scores = {k: np.exp(v - max_score) for k, v in raw_scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}


def compute_confidence(
    outcome_probs: Dict[str, float],
    trajectory_violations: List[Violation]
) -> float:
    """
    confidence = entropy_factor * violation_penalty
    entropy_factor: 1.0 если один исход доминирует, ближе к 0 при равномерном распределении
    violation_penalty: снижает confidence при наличии нарушений
    """
    n = len(outcome_probs)
    H = -sum(p * np.log(p + 1e-10) for p in outcome_probs.values())
    H_max = np.log(n)
    entropy_factor = 1.0 - (H / H_max) if H_max > 0 else 1.0

    soft_violations = sum(1 for v in trajectory_violations if v.severity == "soft")
    violation_penalty = max(0.0, 1.0 - 0.05 * soft_violations)

    return round(entropy_factor * violation_penalty, 4)
```

---

## 10. Модуль core/multiworld.py — мультидоменная композиция

```python
@dataclass
class SharedResourceBus:
    """Единое пространство для ресурсов, разделяемых между доменами."""
    resources: Dict[str, Resource]

    def read(self, resource_id: str) -> float:
        return self.resources[resource_id].value

    def write(self, resource_id: str, value: float):
        self.resources[resource_id].value = value


class MultiDomainWorld:
    """
    Несколько доменных симуляторов, связанных через shared resources.
    Каждый домен видит глобальные ресурсы через SharedResourceBus.
    sign-check автоматически распространяется через границу доменов.
    """
    def __init__(
        self,
        domains: List[WorldState],
        shared_resource_ids: List[str]
    ):
        self.domains = domains
        self.shared_bus = SharedResourceBus(
            resources={
                r_id: self._find_resource(r_id)
                for r_id in shared_resource_ids
            }
        )

    def step(self, policies: Dict[str, List[Policy]]) -> "MultiDomainSimResult":
        all_violations = []
        for domain in self.domains:
            domain_policies = policies.get(domain.domain_id, [])
            # перед шагом: читаем shared resources из bus
            self._sync_from_bus(domain)
            next_domain, violations = step_world(domain, domain_policies)
            # после шага: пишем shared resources обратно в bus
            self._sync_to_bus(next_domain)
            all_violations.extend(violations)

        return MultiDomainSimResult(
            domains=self.domains,
            violations=all_violations,
            shared_state=self.shared_bus.resources
        )
```

---

## 11. Модуль domain/ — схема и компилятор

### 11.1 DomainSchema (JSON-контракт)

Полный пример минимального домена:

```json
{
  "domain_id": "water_market",
  "name": "Water Market Simulation",
  "description": "Regional water resource competition among 5 countries",
  "actors": [
    {
      "id": "country_A",
      "name": "Country A",
      "state": {"military_power": 0.6, "economic_strength": 0.7},
      "metadata": {"population_millions": 45}
    }
  ],
  "resources": [
    {
      "id": "water_stock",
      "name": "Water Stock",
      "value": 1000.0,
      "unit": "km3",
      "evolution_type": "stock_flow",
      "evolution_params": {
        "delta": 0.02,
        "phi_fn": "linear",
        "phi_params": {"base_inflow": 50.0}
      },
      "min_value": 0.0,
      "max_value": 2000.0
    }
  ],
  "relations": [
    {"source_id": "country_A", "target_id": "country_B",
     "relation_type": "water_dependency", "weights": {"dependency": 0.4}}
  ],
  "outcomes": [
    {
      "id": "water_crisis",
      "label": "Regional Water Crisis",
      "scoring_weights": {"water_stock": -2.0, "conflict_level": 1.5},
      "description": "Water stock depleted below critical threshold"
    }
  ],
  "causal_dag": [
    {"source": "water_stock", "target": "agriculture_output", "expected_sign": "+", "strength": "strong"},
    {"source": "conflict_level", "target": "water_stock", "expected_sign": "-", "strength": "strong"}
  ],
  "exogenous_inflows": {
    "water_stock": 50.0
  },
  "metadata": {
    "base_year": 2025,
    "time_unit": "year"
  }
}
```

### 11.2 DomainCompiler

```python
class DomainCompiler:
    def compile(self, schema: Dict) -> WorldState:
        """
        JSON-схема → инициализированный WorldState.
        Валидирует схему, инициализирует операторы, строит causal DAG.
        """
        self._validate_schema(schema)
        world = WorldState(
            domain_id=schema["domain_id"],
            t=0,
            actors={a["id"]: Actor(**a) for a in schema["actors"]},
            resources={r["id"]: Resource(**r) for r in schema["resources"]},
            relations=[Relation(**r) for r in schema["relations"]],
            outcomes={o["id"]: Outcome(**o) for o in schema["outcomes"]},
            causal_dag=[CausalEdge(**e) for e in schema["causal_dag"]],
            metadata=schema.get("metadata", {})
        )
        return world

    def _validate_schema(self, schema: Dict):
        """
        Проверяет:
        - все actor_id в relations существуют в actors
        - все resource_id в causal_dag существуют в resources
        - все evolution_types существуют в EVOLUTION_REGISTRY
        - все scoring_weights в outcomes ссылаются на существующие ключи
        """
        ...
```

### 11.3 GIM15 domain profile

Файл `domain/profiles/gim15.json` — полное описание GIM15 в схеме Freeman.
Это **регрессионный тест**: если GIM15 запускается через Freeman и даёт те же
результаты что и оригинальный GIM15 — реализация корректна.

---

## 12. Модуль game/runner.py

```python
@dataclass
class SimConfig:
    max_steps: int = 50
    dt: float = 1.0
    level2_check_every: int = 5      # sign-consistency каждые N шагов
    convergence_check_steps: int = 20
    convergence_epsilon: float = 1e-4
    fixed_point_max_iter: int = 20
    fixed_point_alpha: float = 0.1
    seed: int = 42

@dataclass
class SimResult:
    domain_id: str
    trajectory: List[WorldState]           # S_0, S_1, ..., S_T
    outcome_probs: List[Dict[str, float]]  # p(o) на каждом шаге
    final_outcome_probs: Dict[str, float]
    confidence: float
    violations: List[Violation]
    converged: bool
    steps_run: int
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        ...

class GameRunner:
    def __init__(self, config: SimConfig):
        self.config = config

    def run(
        self,
        world: WorldState,
        policies: List[Policy],
    ) -> SimResult:
        # 0. level1 check при старте
        l1_violations = level1_check(world, self.config)

        # 1. fixed-point для начальной параметризации
        world, fp_converged, fp_iters = find_fixed_point(
            world, world.causal_dag,
            self.config.fixed_point_max_iter,
            self.config.fixed_point_alpha
        )

        trajectory = [world.clone()]
        all_violations = l1_violations.copy()
        outcome_probs_history = []

        for t in range(self.config.max_steps):
            # шаг
            world, step_violations = step_world(world, policies, self.config.dt)
            all_violations.extend(step_violations)
            trajectory.append(world.clone())

            # scoring
            probs = score_outcomes(world)
            outcome_probs_history.append(probs)

            # level2 rolling check
            if t % self.config.level2_check_every == 0:
                l2 = level2_check(world, world.causal_dag)
                all_violations.extend(l2)

        confidence = compute_confidence(outcome_probs_history[-1], all_violations)

        return SimResult(
            domain_id=world.domain_id,
            trajectory=trajectory,
            outcome_probs=outcome_probs_history,
            final_outcome_probs=outcome_probs_history[-1],
            confidence=confidence,
            violations=all_violations,
            converged=fp_converged,
            steps_run=self.config.max_steps,
            metadata={"fixed_point_iters": fp_iters}
        )
```

---

## 13. Модуль api/tool_api.py — LLM-совместимый API

Freeman должен быть готов к подключению как tool. Четыре функции:

```python
FREEMAN_TOOLS = [
    {
        "name": "freeman_compile_domain",
        "description": "Compile a domain schema into a simulation world. Returns world_id.",
        "parameters": {
            "schema": {"type": "object", "description": "Domain schema JSON"},
        }
    },
    {
        "name": "freeman_run_simulation",
        "description": "Run simulation on a compiled world. Returns SimResult with trajectories and outcome probabilities.",
        "parameters": {
            "world_id": {"type": "string"},
            "policies": {"type": "array"},
            "max_steps": {"type": "integer", "default": 50},
            "seed": {"type": "integer", "default": 42}
        }
    },
    {
        "name": "freeman_get_world_state",
        "description": "Get current state of a world at timestep t.",
        "parameters": {
            "world_id": {"type": "string"},
            "t": {"type": "integer", "default": -1}  # -1 = последний
        }
    },
    {
        "name": "freeman_verify_domain",
        "description": "Run all verification levels on a world. Returns VerificationReport.",
        "parameters": {
            "world_id": {"type": "string"},
            "levels": {"type": "array", "items": {"type": "integer"}, "default": [0, 1, 2]}
        }
    }
]
```

---

## 14. Конфигурация

```yaml
freeman:
  default_evolution: "stock_flow"
  level0_hard_stop: true
  epsilon: 1e-8                    # числовая точность для проверок
  sign_epsilon: 1e-4               # порог для определения знака в level2

sim:
  max_steps: 50
  dt: 1.0
  level2_check_every: 5
  convergence_check_steps: 20
  convergence_epsilon: 1e-4
  fixed_point_max_iter: 20
  fixed_point_alpha: 0.1
  seed: 42

multiworld:
  sync_mode: "after_each_step"     # or "after_all_steps"
```

---

## 15. Порядок разработки

| Этап | Файлы | Критерий готовности |
|---|---|---|
| 1 | `core/types.py` | все dataclass созданы, JSON round-trip работает |
| 2 | `core/world.py` | `snapshot()` / `from_snapshot()` / `clone()` без потерь |
| 3 | `core/evolution.py` | все 5 операторов реализованы, unit tests |
| 4 | `verifier/level0.py` | все 4 проверки, test_level0.py зелёный |
| 5 | `verifier/level1.py` | null convergence + spectral radius, test_level1.py зелёный |
| 6 | `core/scorer.py` | softmax + confidence, simplex всегда = 1.0 |
| 7 | `core/transition.py` | step_world работает, HardStopException при нарушении L0 |
| 8 | `verifier/level2.py` + `fixed_point.py` | sign-check + итерации, test_level2.py зелёный |
| 9 | `domain/schema.py` + `domain/compiler.py` | water_market пример компилируется и запускается |
| 10 | `game/runner.py` | полный цикл на water_market, SimResult сериализуется |
| 11 | `core/multiworld.py` | два домена с shared resource, sync корректен |
| 12 | `domain/profiles/gim15.json` + `tests/test_gim15_compat.py` | GIM15 через Freeman = оригинал |
| 13 | `api/tool_api.py` | все 4 функции вызываются из Python, JSON ответы валидны |
| 14 | интеграционный тест | water_market 50 шагов, нет нарушений L0, confidence > 0 |

---

## 16. Вне скопа v1.0

- MCP server (`api/mcp_server.py`) — v1.1
- Nash / CCE solver (`game/equilibrium.py`) — перенос из GIM15, v1.1
- Агентная логика Freeman-Agent — отдельный репозиторий, после стабилизации v1.0
- Surrogate / emulator model для ускорения дорогих доменов — v2.0
- Fine-tuning LLM на траекториях Freeman — v2.0
