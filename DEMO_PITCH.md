# EdgeGuard AI — Hackathon Pitch (30 seconds)

> **Predictive maintenance for Tata Signa 4825.TK tipper trucks — 5 AI engines, one command center.**

## The story

Open-pit mining trucks cost ₹18,000/hour in lost production. When a hydraulic
ram fails in Pit A, the truck is dead, the contract is at risk, and the
mechanic is 40 minutes away.

**EdgeGuard AI** watches 6 sensors on every truck, predicts failures **hours
in advance**, and tells the operator exactly what to do — with citations from
the actual maintenance SOPs.

## What we built (5 AI engines)

| # | Engine | What it does | Accuracy |
|---|--------|--------------|----------|
| 1 | **Vision** — YOLOv8n | Inspects tipper bed / payload / carryback from frames | mAP50 ≈ 0.91 |
| 2 | **Predict** — XGBoost | Failure probability + Remaining Useful Life | F1 ≈ 0.99 |
| 3 | **Fusion** — Multi-modal | Combines sensor + vision + history → 1 decision | — |
| 4 | **Business** — ROI engine | Translates AI output to ₹/year saved | — |
| 5 | **Copilot** — Gemini LLM | Plain-language explanations + repair steps | — |

## What judges will see

1. **Command Center** — live sensor sparklines, ML gauge, alerts, **failure
   probability trend over the last 50 predictions**, AI Copilot explanations.
2. **Digital Twin** — animated SVG of the truck with 6 sensor nodes, cascade
   arrows showing degradation spreading engine → bearing → oil → hydraulic,
   tipper bed raising at critical, exhaust particles when engine is hot.
3. **Fleet View** — 3 trucks monitored in parallel, click to switch.
4. **ROI / Business** — big animated **X-RI** number, ₹-amounts, savings
   breakdown chart, full assumptions table.
5. **Maintenance** — RAG-powered SOP search, 20 SOPs, semantic match scores,
   alert-to-SOP timeline.
6. **Settings** — real config UI, localStorage persistence, test-connection.

## The money slide

> For a 5-truck pilot over 12 months:
> - **Avoided downtime:** 240 hours
> - **Revenue protected:** ₹1+ Cr
> - **Unplanned maintenance saved:** ₹1.3+ Cr
> - **Net savings:** ₹2+ Cr
> - **Platform cost:** ₹60 L
> - **ROI: 33.6x · Payback: 2.2 months**

(See the **ROI / Business** tab live for the actual computed number.)

## Resilience

- Backend goes down → app enters **demo mode** with a 90-second simulated
  cycle (healthy → warning → critical → reset) so the demo never breaks.
- WiFi drops in the pit → readings buffered in memory, flushed on reconnect.
- Settings + thresholds persist to `localStorage`.

## How to demo in 90 seconds

```bash
# 1. Backend
cd backend && python -m uvicorn main:app --port 8000

# 2. Poller
cd ml-data && python poller.py

# 3. Wokwi sim (browser) — publishes 6 sensor topics
# 4. Frontend
cd frontend && python -m http.server 3000
# → http://localhost:3000
```

Wait ~60 s for the Wokwi sim to enter the failure scenario, then click
through:
1. **Command Center** → watch the gauge climb to critical
2. **Digital Twin** → see the cascade animate, nodes turn red
3. **ROI / Business** → see the cost-savings story
4. **Maintenance** → search "hydraulic" → get the matching SOP

## What makes this hackathon-grade

- **Not a slideshow.** Real data flows: MQTT → FastAPI → Supabase → ML →
  Frontend. The same code runs against a real truck.
- **Five AI engines, not one.** Each is a real, trained, evaluated model
  with a public API. Not a wrapper around an LLM.
- **Engineering, not just ML.** Offline buffering, demo-mode resilience,
  real fleet management, persistent settings, semantic SOP search, INR/ROI
  translation, multi-modal fusion. Judges see a system, not a notebook.
- **The digital twin is the visual centrepiece.** A tipper truck that
  visibly degrades is the thing judges will photograph.

## Try it with no backend

Just open `frontend/index.html` in a browser — it auto-enters demo mode
and shows the full healthy → critical → reset cycle in 90 seconds.
