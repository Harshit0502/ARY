---
title: SupportDesk OpenEnv
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - triage
---

# SupportDesk OpenEnv

SupportDesk OpenEnv is a real-world support-operations simulation for training and evaluating agents on customer ticket triage, billing resolution, and incident coordination.

The environment exposes a standard OpenEnv-style loop through typed actions, typed observations, dense rewards, and deterministic task graders. Each episode presents a realistic case summary with operational constraints and a limited action budget.

## Why this environment exists

Support teams repeatedly perform structured workflows such as:
- routing requests to the correct team
- applying policy-safe resolutions
- drafting compliant customer responses
- escalating incidents without overpromising

This environment turns those workflows into a compact, reproducible benchmark for agent evaluation.

## Tasks

The environment includes three deterministic tasks with increasing difficulty:

1. **Easy — Password-reset login lockout triage**
   - Route the case to the correct support queue
   - assign an appropriate priority
   - provide a safe reply mentioning the reset flow, spam folder, and 2FA guidance

2. **Medium — Duplicate billing charge resolution**
   - route the case to billing
   - classify urgency correctly
   - communicate refund handling through the original payment method and a 3–5 business day processing window

3. **Hard — EU outage incident coordination**
   - assign incident-management ownership
   - classify the outage with the correct urgency and severity
   - write a customer-facing update mentioning the status page and workaround without promising an exact ETA

## Action space

The environment accepts a typed `SupportAction` object.

Main fields:
- `action_type`: `classify | draft_reply | escalate | resolve | finalize`
- `issue_type`
- `priority`: `low | medium | high | urgent`
- `team`
- `severity`: `sev4 | sev3 | sev2 | sev1`
- `status`
- `tags`
- `message`
- `internal_note`
- `refund_amount`
- `confidence`

## Observation space

Each `SupportObservation` includes:
- task id, title, and difficulty
- turn number and remaining turns
- inbox summary
- open questions
- constraints
- last feedback
- current best score
- done flag
- suggested fields for the next action

## Reward design

Rewards are dense and deterministic.

Scoring includes:
- correct routing and classification
- correct resolution fields
- required policy content
- penalties for unsafe or forbidden language
- incremental reward based on improvement over the best prior score

This allows partial progress to be rewarded instead of relying only on final success/failure.

## API

The server exposes these routes:

- `GET /health` — liveness check
- `GET /tasks` — public task metadata only
- `POST /reset` — start a new episode
- `POST /step` — submit a `SupportAction`
- `GET /state` — inspect current episode state (redacted, non-cheating)

## Local setup

Install the package:

```bash
pip install -e ".[dev]"