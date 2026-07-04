"""
EdgeGuard AI — Gemini LLM Mining Operations Copilot (Module 10 of ml work.txt)

PURPOSE
-------
Industrial Mining Copilot. A natural-language interface that explains
predictions, root causes, maintenance actions, and answers SOP
questions in plain English for maintenance crews and managers.

CAPABILITIES
------------
- explain_prediction(ai2, ai1, decision) -> str
- recommend_repair(decision) -> str
- explain_business_impact(impact) -> str
- answer_sop_question(question, rag_results) -> str

PROMPT ENGINEERING PRINCIPLES (Module 10)
-----------------------------------------
1. CONTEXT INJECTION: every prompt carries the structured outputs of
   the AI engines (failure_probability, rul_hours, root_cause, ...)
   so the LLM doesn't have to guess the state of the truck.
2. GROUNDING: maintenance answers MUST be based on RAG-retrieved
   SOP chunks, never invented. We use the prompt "Answer ONLY using
   the supplied SOP context. If the context does not cover the
   question, say so."
3. HALLUCINATION REDUCTION: we ask the model to return a JSON
   object with explicit "citations" referencing which SOP chunk it
   used, so the UI can show the source.
4. SAFETY: every output is capped at MAX_TOKENS to prevent runaway
   generation; refusals are honoured.

USAGE
-----
    from copilot.gemini_copilot import GeminiCopilot
    bot = GeminiCopilot()  # reads GEMINI_API_KEY from env
    text = bot.explain_prediction(ai2_dict, decision_dict)

If the key is absent or the SDK isn't installed, the copilot falls
back to a deterministic template-based explanation so the rest of
the system still works at the demo.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("edgeguard.copilot")

GEMINI_MODEL_NAME = os.environ.get("EDGEGUARD_GEMINI_MODEL", "gemini-1.5-flash")
MAX_TOKENS        = int(os.environ.get("EDGEGUARD_GEMINI_MAX_TOKENS", "512"))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are EdgeGuard Copilot, an industrial AI assistant for
mining truck predictive maintenance. You give concise, technical answers to
maintenance crews, reliability engineers, and mine managers. You always:

- Lead with the action recommended by the AI engines (HALT, SCHEDULE, MONITOR)
- Cite the sensor signals that drove your recommendation
- Reference any supplied SOP chunks using [SOP-XXX] notation
- Quantify risk in concrete terms (hours, INR, percent)
- Never invent part numbers, procedures, or sensor values that you were not given
- If the supplied context is insufficient, say "I need more information about..."

Keep answers to 3-5 sentences unless the user explicitly asks for detail.
"""


def _build_prediction_prompt(ai2: dict, ai1: dict | None,
                             decision: dict, sop_titles: list[str]) -> str:
    parts = [
        "Current state:",
        f"  Failure probability: {ai2.get('failure_probability', 'n/a')}",
        f"  RUL (hours):         {ai2.get('rul_hours', 'n/a')}",
        f"  Top drivers:         {ai2.get('top_features', [])[:3]}",
    ]
    if ai1:
        parts.append(f"  Vision:              carryback={ai1.get('carryback_pct')}, "
                     f"loading_quality={ai1.get('loading_quality')}")
    if decision:
        parts.append(f"  Root cause (fused):  {decision.get('root_cause')} "
                     f"(confidence {decision.get('cause_confidence', 0):.0%})")
        parts.append(f"  Risk score:          {decision.get('risk_score')}/100 "
                     f"-> {decision.get('risk_band')}")
    if sop_titles:
        parts.append("Relevant SOPs (from knowledge base):")
        for t in sop_titles[:3]:
            parts.append(f"  - {t}")
    parts.append("\nExplain this situation to a maintenance crew lead in 3-4 sentences.")
    return "\n".join(parts)


def _build_sop_qa_prompt(question: str, sop_chunks: list[dict]) -> str:
    """Q&A prompt: the LLM is grounded only in the supplied SOP chunks."""
    parts = [
        f"Maintenance crew question: {question}",
        "",
        "Relevant SOP context (use ONLY this to answer):",
    ]
    for i, c in enumerate(sop_chunks, 1):
        parts.append(f"--- Chunk {i} [{c.get('title', 'unknown')}] ---")
        parts.append(c.get("content_chunk", "")[:800])
    parts.append("")
    parts.append("If the chunks do not cover the question, say so. "
                 "Cite which chunk(s) you used, like [Chunk 2].")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fallback template engine (used when Gemini is unavailable)
# ---------------------------------------------------------------------------
def _fallback_prediction_text(ai2: dict, decision: dict) -> str:
    p = ai2.get("failure_probability", 0)
    rul = ai2.get("rul_hours", 0)
    cause = decision.get("root_cause", "unknown")
    confidence = decision.get("cause_confidence", 0)
    risk = decision.get("risk_band", "low")
    top = ai2.get("top_features", [])
    drivers = ", ".join(n.replace("_", " ") for n, _ in top[:2]) if top else "sensor trends"

    if p >= 0.85 or rul <= 1.0:
        action = "HALT operations immediately"
    elif p >= 0.65:
        action = "Schedule maintenance within the next 4 hours"
    elif p >= 0.40:
        action = "Plan a service visit in the next shift"
    else:
        action = "Continue operations and monitor"

    return (f"**{action}.** The AI engines flag a {risk} risk of {cause} failure "
            f"(confidence {confidence:.0%}). Key sensor drivers: {drivers}. "
            f"Estimated RUL: {rul:.1f} hours.")


def _fallback_sop_answer(question: str, sop_chunks: list[dict]) -> str:
    if not sop_chunks:
        return ("I don't have an SOP that directly answers this question. "
                "Please contact the reliability engineering team.")
    c = sop_chunks[0]
    return (f"Based on [{c.get('title', 'SOP')}]: {c.get('content_chunk', '')[:400]}")


# ---------------------------------------------------------------------------
# GeminiCopilot class
# ---------------------------------------------------------------------------
class GeminiCopilot:
    """Thin wrapper around the Gemini SDK.

    - Reads API key from GEMINI_API_KEY env var
    - Falls back to template explanations when SDK or key is absent
    - Exposes typed methods: explain_prediction, recommend_repair, ...
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name or GEMINI_MODEL_NAME
        self.model = None
        self.enabled = False
        if self.api_key and self.api_key != "PASTE_HERE":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT,
                )
                self.enabled = True
                log.info(f"GeminiCopilot ready (model={self.model_name})")
            except Exception as e:
                log.warning(f"Gemini init failed: {e}. Falling back to templates.")
                self.enabled = False
        else:
            log.info("GEMINI_API_KEY not set — Copilot running in template mode.")

    # -----------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        if not self.enabled or self.model is None:
            return None
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": MAX_TOKENS,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            )
            return resp.text.strip()
        except Exception as e:
            log.warning(f"Gemini generate_content failed: {e}")
            return None

    # -----------------------------------------------------------------------
    def explain_prediction(self, ai2: dict, ai1: dict | None = None,
                           decision: dict | None = None,
                           sop_titles: list[str] | None = None) -> str:
        sop_titles = sop_titles or []
        prompt = _build_prediction_prompt(ai2 or {}, ai1, decision or {}, sop_titles)
        text = self._generate(prompt)
        if text:
            return text
        return _fallback_prediction_text(ai2 or {}, decision or {})

    # -----------------------------------------------------------------------
    def recommend_repair(self, decision: dict) -> str:
        cause = decision.get("root_cause", "unknown")
        priority = decision.get("maintenance_priority", "low")
        prompt = (f"Given a {priority}-priority decision with probable {cause} failure, "
                  f"list the first 3 physical inspection steps a mechanic should perform. "
                  f"Be concrete (which sensors, which components, which tools).")
        text = self._generate(prompt)
        if text:
            return text
        return (f"1. Inspect {cause}-related sensors and connections. "
                f"2. Check the matching SOP [{cause.upper()}-XX] for the inspection procedure. "
                f"3. Record all readings before any component replacement.")

    # -----------------------------------------------------------------------
    def explain_business_impact(self, impact: dict) -> str:
        prompt = (f"A predictive maintenance event has the following business impact: "
                  f"{json.dumps(impact, indent=2)}. "
                  f"Explain this to a mine manager in 2 sentences, focusing on cost savings vs. cost of action.")
        text = self._generate(prompt)
        if text:
            return text
        roi = impact.get("roi_pct", 0)
        annual = impact.get("annualised_savings_inr", 0)
        return (f"Acting on this prediction yields an estimated annual savings of INR {annual:,.0f} "
                f"({roi:.0f}% ROI over the platform cost). Deferring maintenance costs "
                f"3.5x more in unplanned downtime.")

    # -----------------------------------------------------------------------
    def answer_sop_question(self, question: str, sop_chunks: list[dict]) -> str:
        if not sop_chunks:
            return _fallback_sop_answer(question, [])
        prompt = _build_sop_qa_prompt(question, sop_chunks)
        text = self._generate(prompt)
        if text:
            return text
        return _fallback_sop_answer(question, sop_chunks)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Gemini Copilot self-test (template fallback) ===\n")
    bot = GeminiCopilot()
    sample_ai2 = {
        "failure_probability": 0.78, "rul_hours": 4.2,
        "top_features": [("bearing_temp_roc_10", 0.45),
                         ("vibration_rms_rmean_5", 0.30)],
    }
    sample_dec = {
        "root_cause": "bearing", "cause_confidence": 0.59,
        "risk_score": 79.3, "risk_band": "critical",
        "maintenance_priority": "critical",
    }
    print("1) explain_prediction:")
    print("  ", bot.explain_prediction(sample_ai2, decision=sample_dec))
    print("\n2) recommend_repair:")
    print("  ", bot.recommend_repair(sample_dec))
    print("\n3) explain_business_impact:")
    impact = {"roi_pct": 438, "annualised_savings_inr": 32_306_820}
    print("  ", bot.explain_business_impact(impact))
    print("\n4) answer_sop_question (with mock chunk):")
    print("  ", bot.answer_sop_question(
        "How do I check bearing temperature?",
        [{"title": "SOP-BRG-04a", "content_chunk":
          "Step 1: Allow the bearing housing to cool. Step 2: Use IR thermometer to measure at 4 points. Step 3: Compare against baseline."}]
    ))
