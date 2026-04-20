# UML — Wind Turbine RUL Prediction

> 작성일: 2026-04-20
> 렌더링: GitHub 네이티브 Mermaid

아키텍처의 구조·동작·배치를 UML 관점으로 정리한다. 본 문서는
`docs/architecture.md` 의 보완 자료이다.

---

## 1. 컴포넌트 다이어그램

시스템을 구성하는 런타임 단위와 그 의존 관계를 나타낸다.

```mermaid
flowchart TB
  Browser[[Web Browser]]

  subgraph FE[Vite Dev Server :5173]
    direction TB
    SPA[React SPA<br/>Dashboard]
    QC[QueryClient<br/>TanStack Query]
    PRX[[HTTP Proxy<br/>/api, /healthz]]
    SPA --> QC
    QC --> PRX
  end

  subgraph BE[FastAPI :8000]
    direction TB
    ROUTERS[APIRouter<br/>pipelines, jobs]
    SCHEMA[Pydantic Schemas]
    SVC_REG[JobRegistry]
    SVC_PL[pipeline_service]
    BG[BackgroundTasks]
    ROUTERS --> SCHEMA
    ROUTERS --> SVC_REG
    ROUTERS --> BG
    BG --> SVC_PL
    SVC_PL --> SVC_REG
  end

  subgraph CORE[ML Core — backend/*.py]
    direction TB
    PIPE[SimpleRULPipeline]
    DL[DataLoader]
    FE_MOD[FeatureExtractor<br/>Selector, Reducer]
    EXP[ExponentialDegradationModel]
    LSTM[LSTMRULPredictor]
    PIPE --> DL
    PIPE --> FE_MOD
    PIPE --> EXP
    PIPE --> LSTM
  end

  FS[(Filesystem<br/>MathWorks .mat,<br/>plots/, results/)]

  Browser -->|HTTP| FE
  PRX -->|HTTP| BE
  SVC_PL -->|import| CORE
  DL --> FS
```

---

## 2. 백엔드 클래스 다이어그램

FastAPI 어댑터와 ML 코어의 주요 클래스. 역할별로 패키지가 명확히 분리된다.

```mermaid
classDiagram
  class Settings {
    +str project_name
    +str version
    +str api_v1_prefix
    +List~str~ cors_origins
  }

  class Job {
    +str id
    +str status
    +str|None step
    +float progress
    +List~str~ logs
    +str|None error
    +dict|None result
  }

  class JobRegistry {
    -Dict~str, Job~ _jobs
    -Lock _lock
    +create() Job
    +get(job_id) Job
    +update(job_id, **fields) void
    +append_log(job_id, msg) void
  }

  class PipelineService {
    <<module>>
    +run_pipeline_job(job_id, registry, use_lstm, seq_len, seed) void
    -_collect_result(pipeline) dict
  }

  class SimpleRULPipeline {
    +dict components
    +bool use_lstm
    +int lstm_sequence_length
    +DataFrame data
    +ndarray features
    +ndarray health_indicator
    +ndarray true_rul
    +dict results
    +step1_load_data() bool
    +step2_extract_features() bool
    +step3_prepare_targets() bool
    +step4_train_exponential_model() bool
    +step5_train_lstm_model() bool
    +run_pipeline() bool
  }

  class DataLoader {
    +str data_folder
    +int sampling_frequency
    +download_wind_turbine_data() bool
    +load_mat_file(path) dict
    +load_wind_turbine_data() DataFrame
    +load_simulated_data() DataFrame
  }

  class FeatureExtractor {
    +extract_features_from_dataframe(df) DataFrame
    +apply_smoothing(df) DataFrame
  }

  class FeatureSelector {
    +select_features_by_monotonicity(df) List~str~
  }

  class DimensionReducer {
    +fit_transform(df) ndarray
    +transform(df) ndarray
    +get_health_indicator(pca) ndarray
  }

  class ExponentialDegradationModel {
    +float theta_posterior
    +float beta_posterior
    +float noise_variance
    +train(features, rul, hi) void
    +predict(features, hi) ndarray
    +predict_rul(hi) tuple
  }

  class LSTMRULPredictor {
    +int sequence_length
    +int lstm_units
    +int epochs
    +int random_seed
    +train(features, rul) void
    +predict(features) ndarray
    -prepare_data(features, rul) tuple
  }

  PipelineService ..> JobRegistry : updates
  PipelineService ..> SimpleRULPipeline : runs
  SimpleRULPipeline --> DataLoader
  SimpleRULPipeline --> FeatureExtractor
  SimpleRULPipeline --> FeatureSelector
  SimpleRULPipeline --> DimensionReducer
  SimpleRULPipeline --> ExponentialDegradationModel
  SimpleRULPipeline --> LSTMRULPredictor
  JobRegistry "1" o-- "*" Job
```

---

## 3. 프론트엔드 컴포넌트 트리

리액트 컴포넌트와 데이터 소스의 관계.

```mermaid
classDiagram
  class App {
    +render() JSX
  }

  class Dashboard {
    -string|null jobId
    -bool useLstm
    -UseMutationResult startRun
    -UseQueryResult jobQuery
    +render() JSX
  }

  class RunControls {
    +bool useLstm
    +bool running
    +onRun() void
  }

  class JobProgress {
    +JobStatus status
  }

  class Results {
    +string jobId
    -UseQueryResult summary
    -UseQueryResult hi
    -UseQueryResult preds
    -UseQueryResult metrics
  }

  class PlotlyChart {
    +Data[] data
    +Partial~Layout~ layout
    +number height
  }

  class ApiClient {
    <<module>>
    +runPipeline(req) Promise
    +getJobStatus(id) Promise
    +getDatasetSummary(id) Promise
    +getHealthIndicator(id) Promise
    +getPredictions(id) Promise
    +getMetrics(id) Promise
  }

  App --> Dashboard
  Dashboard --> RunControls
  Dashboard --> JobProgress
  Dashboard --> Results
  Results --> PlotlyChart
  Dashboard ..> ApiClient : uses
  Results ..> ApiClient : uses
```

---

## 4. 시퀀스 다이어그램 — 파이프라인 실행

"Run pipeline" 버튼부터 결과 렌더까지의 메시지 흐름.

```mermaid
sequenceDiagram
  autonumber
  actor U as User
  participant D as Dashboard<br/>(React)
  participant V as Vite Proxy<br/>(:5173)
  participant R as Router<br/>(pipelines.py)
  participant JR as JobRegistry
  participant BG as BackgroundTasks
  participant PS as pipeline_service
  participant PL as SimpleRULPipeline

  U->>D: click "Run pipeline"
  D->>V: POST /api/v1/pipelines/run {use_lstm, seed}
  V->>R: forward
  R->>JR: create() → job_id
  R->>BG: add_task(run_pipeline_job, ...)
  R-->>D: 202 {job_id, status:"pending"}

  par Poll status
    loop every 1.5s until terminal
      D->>V: GET /api/v1/jobs/{id}
      V->>R: forward
      R->>JR: get(id)
      JR-->>R: Job snapshot
      R-->>D: JobStatus
    end
  and Background execution
    BG->>PS: run_pipeline_job(id)
    PS->>JR: update(status="running")
    PS->>PL: step1..step5
    loop each step
      PS->>JR: update(step, progress)
      PS->>JR: append_log(...)
    end
    PS->>PS: _collect_result()
    PS->>JR: update(status="succeeded", result)
  end

  Note over D: status == "succeeded"
  par Fetch results (parallel useQuery)
    D->>V: GET /pipelines/{id}/dataset-summary
    D->>V: GET /pipelines/{id}/health-indicator
    D->>V: GET /pipelines/{id}/predictions
    D->>V: GET /pipelines/{id}/metrics
  end
  V->>R: forward each
  R->>JR: get(id).result
  R-->>D: typed JSON
  D-->>U: render charts + metrics table
```

---

## 5. 상태 다이어그램 — Job 생명주기

`JobRegistry` 가 추적하는 단일 Job 의 상태 전이.

```mermaid
stateDiagram-v2
  [*] --> pending : create()
  pending --> running : run_pipeline_job 시작
  running --> running : update(step, progress)
  running --> succeeded : _collect_result() 저장
  running --> failed : 예외 or step() == False
  succeeded --> [*]
  failed --> [*]

  note right of running
    step ∈ { load_data, extract_features,
             prepare_targets, exponential_model,
             lstm_model }
  end note
```

---

## 6. 패키지 / 모듈 의존성

상위 레벨 import 방향. 화살표는 "import 함"을 의미한다.

```mermaid
flowchart LR
  subgraph app[app/]
    app_main[main.py<br/>create_app]
    api_pipe[api/pipelines.py]
    api_jobs[api/jobs.py]
    schemas[schemas/pipeline.py]
    svc_reg[services/job_registry.py]
    svc_pl[services/pipeline_service.py]
    core_set[core/settings.py]
  end

  subgraph core[backend ML 코어]
    m_main[main.py<br/>SimpleRULPipeline]
    m_models[models.py]
    m_feat[feature_extractor.py]
    m_data[data_loader.py]
    m_cfg[config.py]
    m_utils[utils.py]
  end

  app_main --> api_pipe
  app_main --> api_jobs
  app_main --> core_set
  api_pipe --> schemas
  api_pipe --> svc_reg
  api_pipe --> svc_pl
  api_jobs --> schemas
  api_jobs --> svc_reg
  svc_pl --> svc_reg
  svc_pl --> m_main
  svc_pl --> m_utils
  svc_pl --> m_cfg
  m_main --> m_models
  m_main --> m_feat
  m_main --> m_data
  m_main --> m_cfg
  m_data --> m_cfg
  m_models --> m_cfg
  m_feat --> m_cfg
```

> 규칙: `app/` 은 ML 코어에 의존하지만 역방향은 없다. ML 코어만 따로 떼어
> CLI(`python main.py`) 또는 다른 프런트(Streamlit, CLI 툴) 에서도 재사용 가능.
