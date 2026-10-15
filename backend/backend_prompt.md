# Backend / Sports Intelligence — Architecture & Implementation Prompt

## 1. Project Context

This repository is the existing Python project for a football/sports analysis system.

The project already contains AI/ML code and inference pipelines. It may contain models and scripts for event spotting, player detection, player tracking, ball detection, team affiliation, player/team identification, and other football analysis tasks.

The frontend is being developed separately in:

`../frontend_Sports-Classification`

The frontend will eventually communicate with this Python project through a backend/API.

The long-term vision is a Football Intelligence / Sports Intelligence platform.

```text
                         VIDEO
                           │
                           ▼
                ┌─────────────────────┐
                │ Football Analysis   │
                │ Engine              │
                │                     │
                │ Detection           │
                │ Tracking            │
                │ Team affiliation    │
                │ Event spotting      │
                │ Spatial analysis    │
                │ Tactical analysis   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Football Memory     │
                │                     │
                │ Players             │
                │ Teams               │
                │ Events              │
                │ Tracking            │
                │ Statistics          │
                │ Match context       │
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             PostgreSQL          Vector DB
                 │
                 ▼
              FastAPI
                 │
                 ▼
        React / TypeScript frontend
                 │
                 ▼
          Future AI Agent / LLM
```

This is the long-term architecture.

However, the current task is NOT to implement the entire system.

---

# 2. Most Important Rule

Before modifying anything:

**Inspect the existing project thoroughly.**

The existing Python project is the source of truth for:

- models
- classes
- inference pipelines
- output structures
- JSON formats
- detections
- tracking
- events
- timestamps
- coordinates
- player IDs
- team IDs
- confidence values
- video processing
- configuration
- existing utilities

Do not invent these structures if they already exist.

Do not rewrite working AI code unnecessarily.

Do not duplicate AI logic.

The backend should wrap and expose the existing analysis capabilities rather than replacing them.

---

# 3. First Task — Analysis

For the first task, inspect the entire repository before implementing the backend.

Understand:

### Project structure

Identify:

- entry points
- scripts
- model code
- inference code
- configuration
- utilities
- generated outputs
- datasets
- example files
- existing APIs, if any
- dependencies
- how inference is currently executed

### AI pipeline

Determine:

1. What models exist?
2. How are they loaded?
3. What inputs do they require?
4. What outputs do they produce?
5. How are outputs represented?
6. Which components can already be reused?
7. Which components are tightly coupled?
8. Which components should remain untouched?

### Output structures

Find the real structures produced by the current pipeline.

For example, determine whether outputs contain information such as:

- frame number
- timestamp
- bounding boxes
- player ID
- team
- confidence
- ball position
- event type
- event timestamp
- event confidence
- tracking trajectory
- video metadata

Do NOT assume these fields exist.

Inspect the actual code and output files.

---

# 4. Relationship With Frontend

The frontend lives at:

```text
../frontend_Sports-Classification
```

Inspect it enough to understand the intended consumer of the backend.

The frontend is React + TypeScript + Vite.

The backend should eventually provide structured data that the frontend can consume.

However:

**Do not invent an API contract simply to make the frontend work.**

First understand the Python outputs.

Then propose a clean API/domain representation that can act as a stable contract between Python analysis and React.

---

# 5. Backend Technology

The intended backend technology is:

- Python
- FastAPI
- Pydantic
- PostgreSQL eventually

Potential future components:

- SQLAlchemy
- Alembic
- PostgreSQL
- pgvector or another vector database if justified later
- background job processing if necessary

But do NOT add all of these immediately.

Choose dependencies only when there is a concrete reason.

The first implementation should remain simple.

---

# 6. Important Architectural Principle

Separate these concepts:

```text
AI / Analysis Engine
        ↓
Normalization / Domain Layer
        ↓
Football Memory
        ↓
Persistence
        ↓
API
        ↓
Frontend
```

Do not make FastAPI endpoints directly dependent on arbitrary model-specific output structures.

Prefer:

```text
T-DEED / existing model
        ↓
adapter
        ↓
Football Event / domain object
        ↓
Football Memory
        ↓
FastAPI
        ↓
React
```

This allows models to change without breaking the frontend.

---

# 7. Football Memory

Football Memory is an important architectural concept in this project.

It should eventually represent the structured knowledge of a football match.

Conceptually it may contain:

```text
Match
 ├── metadata
 ├── teams
 │    ├── players
 │    └── statistics
 ├── events
 │    ├── goals
 │    ├── shots
 │    ├── cards
 │    ├── corners
 │    └── ...
 ├── tracking
 │    ├── player trajectories
 │    └── ball trajectory
 ├── spatial information
 ├── tactical information
 └── derived statistics
```

But this is conceptual only.

Do not implement fields that are not supported by the existing project yet.

The first version of Football Memory should be based on real available data.

---

# 8. Analytics vs Football Memory

Keep these concepts separate.

### Analysis / Analytics

Computes derived information such as:

- distance covered
- speed
- possession
- heatmaps
- passing networks
- tactical metrics
- spatial statistics

### Football Memory

Stores and organizes the resulting structured knowledge so that it can be:

- queried
- persisted
- visualized
- retrieved
- used by future agents/LLMs

Conceptually:

```text
Model outputs
      ↓
Analytics
      ↓
Football Memory
      ↓
API / Frontend / Agent
```

Do not implement advanced analytics unless the existing project already supports them or they are explicitly requested later.

---

# 9. API Vision

The eventual API may contain resources such as:

```text
GET    /matches
POST   /matches
GET    /matches/{match_id}

GET    /matches/{match_id}/events
GET    /matches/{match_id}/players
GET    /matches/{match_id}/tracking
GET    /matches/{match_id}/statistics

POST   /matches/{match_id}/analysis
GET    /matches/{match_id}/analysis/status
```

These are examples only.

DO NOT implement these endpoints blindly.

First determine what the existing Python system can actually provide.

Then propose the API contract.

---

# 10. Match Analysis Lifecycle

The long-term system should conceptually support:

```text
1. User uploads match video
        ↓
2. Match is created
        ↓
3. Analysis starts
        ↓
4. AI pipeline processes video
        ↓
5. Results are normalized
        ↓
6. Football Memory is generated
        ↓
7. Results are persisted
        ↓
8. Frontend retrieves match data
```

Potential statuses:

```text
PENDING
PROCESSING
COMPLETED
ERROR
```

Do not implement this lifecycle until the existing pipeline has been understood.

---

# 11. Video Processing

The backend will eventually need to handle match videos.

Investigate:

- Where videos are currently stored.
- How videos are passed to inference.
- Whether the pipeline expects a file path.
- Whether it processes local files.
- Whether it generates annotated videos.
- Whether it generates JSON.
- Whether it generates frame-level data.
- How long inference takes.
- Whether processing is synchronous or asynchronous.

The backend should eventually orchestrate the existing pipeline rather than duplicate it.

---

# 12. Video Upload

The first backend implementation should include real video upload support.

Create a dedicated endpoint such as:

```text
POST /api/videos/upload
```

Use FastAPI's file upload mechanism.

The endpoint should:

1. Accept a video file.
2. Validate that the uploaded file is a supported video format.
3. Generate a unique video identifier.
4. Store the original video in a dedicated storage directory.
5. Return metadata about the uploaded video.
6. Keep storage logic separate from the API route.

Example only:

```json
{
  "video_id": "abc123",
  "filename": "match.mp4",
  "status": "uploaded"
}
```

Do not hardcode this exact response if the real domain model suggests a better representation.

Do not invent metadata.

---

# 13. Video Storage

Do not put uploaded videos directly inside the API route.

Create a dedicated storage/service abstraction.

Conceptually:

```text
POST /videos/upload
        ↓
Video Router
        ↓
Video Service
        ↓
Video Storage
        ↓
stored video
```

The initial implementation can use the local filesystem.

For example:

```text
backend/
└── storage/
    └── videos/
        ├── abc123/
        │   └── match.mp4
        └── ...
```

The architecture should later allow the storage implementation to be replaced without changing the API routes.

Do not introduce cloud storage unless explicitly requested.

---

# 14. Video Metadata

Where possible, store basic metadata such as:

- video ID
- original filename
- stored filename/path
- file size
- format
- upload timestamp
- duration, if it can be obtained reliably
- processing status

Do not implement metadata that cannot be reliably obtained.

---

# 15. Video Retrieval / Streaming

The frontend should eventually be able to load the uploaded match video from the backend.

Provide an appropriate endpoint for video retrieval/streaming, for example:

```text
GET /api/videos/{video_id}
```

Use FastAPI's appropriate file/streaming mechanisms.

The implementation should support browser video playback, including HTTP range requests if needed by the chosen approach.

Do not assume that returning the whole file in a JSON response is sufficient.

---

# 16. Match Endpoints

Implement an initial match API based on the domain models discovered during repository analysis.

Potential endpoints:

```text
GET /api/matches
GET /api/matches/{match_id}
```

If there is not yet a database, use a simple temporary/in-memory or file-based representation.

Do NOT introduce PostgreSQL just to implement these first endpoints.

The architecture should make it possible to replace the temporary repository with PostgreSQL later.

---

# 17. Analysis Integration

The backend should eventually be able to start the existing AI pipeline for an uploaded video.

Conceptually:

```text
POST /api/matches/{match_id}/analysis
        ↓
Analysis Service
        ↓
Existing Python AI pipeline
        ↓
Raw model outputs
        ↓
Adapters
        ↓
Football Memory
```

For the first implementation, do not rewrite the AI pipeline.

Instead, identify the existing entry points/scripts/functions that perform inference and create a service layer capable of calling them.

If the current pipeline is not yet structured enough to be called safely from a service, document what needs to be refactored.

Do not perform a large AI refactor without approval.

---

# 18. Analysis Status

The backend should eventually expose analysis status.

Potential endpoint:

```text
GET /api/matches/{match_id}/analysis/status
```

Potential states:

```text
PENDING
PROCESSING
COMPLETED
ERROR
```

Do not introduce a complex job queue yet unless the existing inference requirements make it necessary.

For the initial implementation, a simple service/state mechanism is acceptable.

---

# 19. Separation of Responsibilities

Keep responsibilities separated:

```text
FastAPI Router
      ↓
Service
      ↓
Adapter
      ↓
Existing AI pipeline
      ↓
Output
```

Do NOT put inference logic directly inside FastAPI route functions.

Avoid:

```python
@app.post("/analysis")
def analysis(...):
    # hundreds of lines of model inference
```

Prefer:

```python
@app.post("/analysis")
def analysis(...):
    return analysis_service.start(...)
```

---

# 20. API Contract for Frontend

The backend will eventually be consumed by:

```text
frontend_Sports-Classification
```

Therefore:

- responses should be predictable
- schemas should be explicit
- errors should be meaningful
- IDs should be stable
- timestamps should have a consistent representation
- coordinates should use a documented convention
- model-specific details should not leak unnecessarily into the frontend

The backend should become the stable contract between the Python analysis system and React.

---

# 21. CORS

Configure CORS appropriately for local frontend development.

The frontend and backend will initially run separately, for example:

```text
Frontend:
http://localhost:5173

Backend:
http://localhost:8000
```

Do not use an unrestricted production CORS configuration.

Use a development configuration that can later be moved to environment variables.

---

# 22. Environment Configuration

Do not hardcode important configuration.

Prepare a simple configuration mechanism for:

- backend host
- backend port
- storage directory
- allowed frontend origin
- model paths if required

Do not over-engineer configuration at this stage.

---

# 23. API Documentation

FastAPI's automatic OpenAPI/Swagger documentation should work.

Verify that the API can be explored through the generated documentation.

The backend should be easy to test manually during development.

---

# 24. Testing

Add basic tests for the initial API:

- health endpoint
- video upload
- invalid video upload
- video retrieval
- match retrieval

Do not build an extensive test suite yet.

---

# 25. Database Comes Later

PostgreSQL is the intended long-term relational database.

Do NOT introduce PostgreSQL in the first implementation unless the existing project already requires it.

Initially:

```text
FastAPI
  ↓
temporary repository / filesystem
```

Later:

```text
FastAPI
  ↓
Repository interface
  ↓
PostgreSQL
```

This allows us to stabilize the domain and API before committing to the final persistence layer.

Potential future entities include:

```text
matches
teams
players
events
tracking
statistics
analysis_runs
```

Do not create a complete database schema yet.

---

# 26. Files vs Database

Large/high-frequency data such as tracking data may not necessarily belong directly in normal relational rows.

For example:

```text
Match
 ├── metadata → PostgreSQL
 ├── events → PostgreSQL
 ├── players → PostgreSQL
 ├── statistics → PostgreSQL
 └── tracking artifact → file/object storage/database strategy
```

This decision should be based on the real output sizes and access patterns.

Analyze the current outputs before deciding.

---

# 27. Domain Models

The backend should eventually define stable domain models using Pydantic.

Potential conceptual examples:

```python
Match
Team
Player
Event
TrackingPoint
PlayerTrajectory
MatchStatistics
AnalysisStatus
```

These are examples only.

Base the actual models on the existing project.

The API models should represent the football domain rather than expose model-specific implementation details.

---

# 28. Existing AI Configuration

Investigate how the existing project currently manages:

- model paths
- video paths
- output directories
- configuration
- environment variables
- device selection
- GPU/CPU
- thresholds

The backend should eventually centralize runtime configuration without breaking existing scripts.

---

# 29. Error Handling

Eventually the backend should handle:

- invalid video
- unsupported format
- missing model
- inference failure
- malformed model output
- analysis failure
- missing match
- invalid parameters

Errors should be explicit and understandable.

Do not build an elaborate error system during the initial implementation.

---

# 30. Documentation

The backend should eventually include:

- README
- API documentation through FastAPI/OpenAPI
- setup instructions
- environment configuration
- explanation of the analysis pipeline
- explanation of Football Memory

Keep documentation concise and practical.

---

# 31. Future LLM / Agent

A future component may be a Football Agent / LLM.

For example:

```text
User:
"How many kilometers did Barcelona players run compared with Madrid?"

        ↓

Football Agent
        ↓
backend tools / Football Memory
        ↓
structured statistics
        ↓
LLM generates answer
```

The LLM should NOT directly analyze the raw video.

The computer vision and analytics pipeline should produce structured information first.

The LLM is a consumer of Football Memory.

Do not implement this now.

---

# 32. Future Research Direction

The project may later include:

- T-DEED improvements
- contextual event spotting
- SoccerMaster integration
- orchestration
- Football Memory
- video retrieval
- long-video reasoning
- specialized agents
- LLM-based natural language querying

These are future layers.

The current backend should be designed so these components can be integrated later without major restructuring.

---

# 33. Implementation Strategy

After the initial repository analysis and architecture proposal have been reviewed and approved, implement the backend incrementally.

Do not implement the complete final architecture in one step.

## Phase 1 — FastAPI Foundation

Create a dedicated backend application.

Prefer a structure similar to:

```text
Sports-Classification/
│
├── existing AI / ML code
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── matches.py
│   │   │       ├── videos.py
│   │   │       └── analysis.py
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── adapters/
│   │   └── core/
│   │
│   ├── storage/
│   │   ├── videos/
│   │   └── outputs/
│   │
│   └── tests/
│
└── ...
```

Adapt this structure to the real repository rather than forcing it.

## Phase 2 — Health

Implement:

```text
GET /api/health
```

Example:

```json
{
  "status": "ok"
}
```

## Phase 3 — Video Storage

Create a storage abstraction using the local filesystem.

## Phase 4 — Video Upload

Implement:

```text
POST /api/videos/upload
```

Support validation, unique IDs, storage, and metadata.

## Phase 5 — Video Retrieval

Implement an endpoint capable of serving/streaming the uploaded video to the browser.

## Phase 6 — Match Domain

Implement the initial match representation and:

```text
GET /api/matches
GET /api/matches/{match_id}
```

Use a temporary/in-memory/file repository until PostgreSQL is justified.

## Phase 7 — Match + Video

Connect uploaded videos to matches.

## Phase 8 — Analysis Service

Identify the real inference entry points and expose them through a service layer.

Do not duplicate or rewrite model code.

## Phase 9 — Analysis API

Add:

```text
POST /api/matches/{match_id}/analysis
GET /api/matches/{match_id}/analysis/status
```

## Phase 10 — Normalized Results

Expose the actual analysis outputs through stable domain/API models that the frontend can consume.

---

# 34. Incremental Development Rule

After each major phase:

1. Run the application.
2. Verify the endpoints.
3. Run tests.
4. Explain what was implemented.
5. Identify any assumptions.
6. Stop before making large architectural changes.

Do not silently refactor unrelated parts of the project.

---

# 35. Development Rules

Follow these rules:

1. Do not rewrite working AI models.
2. Do not duplicate AI logic.
3. Do not invent outputs.
4. Do not invent API contracts without inspecting the real system.
5. Do not create unnecessary dependencies.
6. Do not over-engineer the MVP.
7. Keep analysis code separated from API code.
8. Keep domain models separated from model-specific outputs.
9. Use adapters between model outputs and domain objects.
10. Make the architecture extensible to other sports eventually.
11. Current implementation should remain football-focused.
12. Do not modify the frontend unless explicitly requested.
13. Do not make assumptions about future LLM functionality.
14. Do not implement the Football Agent yet.
15. Do not implement advanced analytics unless explicitly requested.
16. Prefer simple, maintainable Python code.
17. Preserve existing functionality.
18. Keep video storage and upload logic separate from API routes.
19. Keep FastAPI routes thin and delegate logic to services.
20. Prefer small, reviewable changes over large rewrites.

---

# 36. First Claude Session

For the first session:

1. Inspect the complete `Sports-Classification` repository.
2. Inspect `../frontend_Sports-Classification` enough to understand the future consumer.
3. Analyze the existing AI pipeline and outputs.
4. Identify the actual inference entry points.
5. Analyze the current video/input/output workflow.
6. Propose the backend architecture.
7. Propose the initial API contract.
8. Propose the video upload/storage architecture.
9. Identify any code that must be minimally refactored to expose the AI pipeline safely.
10. Provide an incremental implementation roadmap.

**Do not modify files during this first session.**

Do not create the backend yet.

Do not create a database.

Do not refactor the AI pipeline yet.

Do not create dozens of files.

Do not invent data or API contracts.

After presenting the analysis and proposal, STOP and wait for approval.

---

# 37. Final Objective

The final backend should progressively evolve toward:

```text
                        VIDEO
                          │
                          ▼
                 FastAPI / Upload
                          │
                          ▼
                  Match + Storage
                          │
                          ▼
                Analysis Service
                          │
                          ▼
                 Existing AI/ML
                          │
                          ▼
                     Adapters
                          │
                          ▼
                  Football Memory
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             PostgreSQL       Artifacts
                 │
                 ▼
               FastAPI
                 │
                 ▼
       React / TypeScript frontend
                 │
                 ▼
          Future Football Agent
```

The priority is to build a clean, working bridge between the existing Python AI pipeline and the React frontend while preserving the existing AI work and allowing the architecture to grow incrementally.
