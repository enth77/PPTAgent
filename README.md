# PPTAgent: Multi-LLM Lecture Slide Agent

This repository provides a **multi-LLM agent** blueprint for generating university programming lecture slides using OpenAI, GLM, and Groq APIs. It plans, drafts, reviews, and assembles lecture materials into a structured slide deck with exercises, code examples, and teaching notes.

## Goals

- Produce lecture slides that are pedagogically strong: learning objectives, key concepts, examples, practice, and summary.
- Support multiple LLM providers (OpenAI, GLM, Groq) with fanout and synthesis.
- Provide a reproducible pipeline for content planning, review, and slide generation.

## Architecture Overview

```
User Topic/Week
   │
   ▼
[Planner Agent] ──► Lesson plan (objectives, outline, timebox)
   │
   ├──► [Content Agent]  ──► Explanations + examples + exercises
   │
   ├──► [Code Agent]     ──► Runnable code snippets + annotations
   │
   ├──► [Quiz Agent]     ──► MCQ/short answers + rubrics
   │
   ├──► [Critic Agent]   ──► Pedagogy check + clarity + correctness
   │
   ▼
[Slide Composer] ──► Markdown/JSON slide spec
   │
   ▼
[Renderer] ──► PPTX/HTML/PDF
```

### Key Ideas

- **Multi-LLM routing**: use OpenAI for structure, GLM for language localization, Groq for fast iterations.
- **Critic loop**: multiple review passes for correctness, pedagogy, and alignment with course outcomes.
- **Separation of concerns**: plan → content → review → compose.

## Quick Start

1. Install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables:

```
export OPENAI_API_KEY=...
export GLM_API_KEY=...
export GROQ_API_KEY=...
```

3. Run the pipeline:

```
python scripts/run_pipeline.py \
  --topic "Sorting Algorithms" \
  --level "Undergraduate" \
  --weeks 1 \
  --output outputs/sorting_algorithms \
  --synth-provider openai
```

## Directory Layout

```
.
├── README.md
├── requirements.txt
├── AGENTS.md
└── scripts
    ├── run_pipeline.py
    └── providers.py
```

## Notes

- Outputs are generated under `outputs/<topic_slug>/` with per-stage raw and synthesized markdown.
- The slide spec is stored as `slides.json`, and `slides.md` is produced when JSON output is valid.
- The included pipeline focuses on content quality; PPTX rendering is a separate step.
