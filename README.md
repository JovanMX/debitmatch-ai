# DebitMatch AI

A multi-agent advisor that recommends Mexican debit and payroll accounts. It
starts from the user's balance, priorities and constraints, and checks every
figure against the official catalog and the banks' RECA documents. Capstone
project for the CMU Agentic AI program.

> **Architecture in one line:** a LangGraph squad of five agents (Supervisor,
> Profiling, Research, Evaluator, Advisory) over a typed shared state. The
> Evaluator ranks the accounts, with **Tree-of-Thought** beam search for
> hybrid queries. **RAG** over the RECA (Adhesion Contract Registry) contracts 
> and fee annexes checks the fine print. A deterministic calculator produces
> every MXN figure. The draft passes a critique loop with guardrails, and is 
> escalated to a human reviewer when confidence, amount, data freshness or 
> source conflicts require it.

The code walkthrough and the design-to-code map are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What it does

- **Natural-language profiling.** For example: "I keep about 20,000 pesos and
  want zero fees and some yield."
  - The user is assigned one of four archetypes: Flow Manager, Strategic
    Saver, Basic/Protected User or Lifestyle Maximizer.
  - The query becomes a preference vector `w = [yield, fee, perks]`.
- **Two options, never a single "best" claim:**
  - a **Base Option**, the best $0-fee account;
  - a **Positive Trade-off**, which earns more if the user accepts something.
- **Exact figures from the calculator** for each option: annual net impact,
  fees and after-tax yield.
- **Supporting RECA clauses** for each option.
- **Explicit handling of hard cases:**
  - accounts that pay more at a higher balance;
  - when the top-scoring account is not recommended, the answer says why;
  - stale or conflicting data, which leads to abstention.
- **Multi-turn refinement.** Follow-ups such as "make it digital" or "it is
  for my daughter" update the same profile.
- **Human review.** A reviewer console handles escalated cases.
- **MCP server.** The same tools are available to any MCP client.

Catalog: 118 products from 24 institutions, plus 28 official RECA PDFs.

## Quick start

Requires Python 3.12 and an OpenAI API key.

```powershell
python -m venv .venv
.venv\Scripts\activate                 # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                  # then set OPENAI_API_KEY in .env

python ingest_docs.py                   # builds data/chroma_db (first run only, a few minutes)
streamlit run app.py                    # http://localhost:8501
```

`data/chroma_db` is not versioned, so it has to be built once after cloning.

To see one query end to end in the terminal:

```powershell
python scripts\run_demo.py "I want an account with cashback and lifestyle benefits" --deposit 30000
python scripts\run_demo.py --trace              # also prints the full trajectory log
```

## Other entry points

| Command | What it runs |
|---|---|
| `python mcp_server.py` | MCP server (stdio): product search, net-impact calculator, clause search, full recommendation, conversation tools |
| `streamlit run app.py` → *Reviewer Console* page | Human review of escalated cases |
| `python ingest_docs.py --audit-catalog` | Rebuild the index and print catalog consistency findings |

## Project layout

```text
debitmatch_ai/
├── app.py                    Streamlit UI (profile sidebar, reasoning inspector, dashboard)
├── pages/1_Reviewer_Console.py   human review of escalated cases
├── graph.py                  LangGraph topology, time budget, fallback mode
├── state.py                  shared typed state, freshness rules
├── agents/                   router, profiling, extraction, research, supervisor, evaluator, advisory
├── tools/                    calculator, catalog search, RAG engine, Pydantic schemas
├── prompts/                  system prompts and structured output contracts
├── guardrails.py             abstention, source conflicts, draft review, HITL rules
├── hitl_queue.py             persistent review queue + golden dataset
├── memory.py                 multi-turn conversation memory
├── mcp_server.py             MCP server
├── config.py                 models per task, LangSmith tracing
├── ingest_docs.py            builds the ChromaDB index
├── data/
│   ├── catalog.json          structured product catalog (validated on load)
│   ├── pdfs/                 official RECA documents + document_map.json
│   └── rag_docs/             per-product policy texts
├── evaluation/               benchmark dataset, KPI evaluators, LangSmith runner, sample results
├── scripts/run_demo.py       one query from the terminal
├── tests/                    unit, integration, UI and benchmark tests
└── docs/ARCHITECTURE.md      code walkthrough and design-to-code map
```

## Design documents

The design was written before and during the implementation, in six capstone
documents. The table shows where each one lives in the code.

| Design document | Main concepts | Code |
|---|---|---|
| Scoping and Initial Agent Design | Problem, archetypes, Base Option / Trade-off | `agents/profiling.py`, `agents/advisory.py` |
| Agent Design Refinement | Preference vector, hybrid extraction, typed state, short-term memory | `agents/extraction.py`, `state.py`, `memory.py` |
| Multi-Agent Architecture and Coordination Plan | Five agents, supervisor, time budget, routing | `agents/`, `graph.py` |
| RAG and Retrieval Design Integration | Intent routing, chunking, MMR, metadata filters, atomic benefit chunks | `agents/router.py`, `agents/research.py`, `tools/rag_engine.py` |
| Tree-of-Thought Integration Plan | Beam search, stages, stale down-weighting, tie resolution | `agents/evaluator.py` |
| Safety Guardrails and Human Intervention Plan | Abstention, critique loop, HITL, review queue, fallback | `guardrails.py`, `hitl_queue.py`, `graph.py` |

## Tests

```powershell
python tests\test_infra.py           # catalog contract, calculator, RAG, ingestion (12)
python tests\test_agent_graph.py     # end-to-end graph scenarios (5)
python tests\test_doc_alignment.py   # requirements from the design documents (26)
python tests\test_evaluators.py      # evaluators catch injected faults (8)
python tests\test_ui_components.py   # Streamlit AppTest, deterministic mode (27)
python tests\test_llm_connection.py  # smoke test of the configured OpenAI models
```

The functional suites skip the chat models, which keeps them deterministic.
They still need the `data/chroma_db` index and the OpenAI key, which is used
for query embeddings.

## Evaluation

```powershell
python tests\test_e2e_benchmark.py --dataset_size 15 --mode local
python tests\test_e2e_benchmark.py --mode langsmith       # needs LANGCHAIN_API_KEY
```

Latest run (15 cases, `evaluation/sample_results/`):

| KPI | Result | Target | |
|---|---|---|---|
| RAG Triad faithfulness | 0.990 | ≥ 0.95 | ✅ |
| Exact-match arithmetic | 100 % | 100 % | ✅ |
| Freshness safety violations | 0 % | 0 % | ✅ |
| Expected Calibration Error | 0.113 | < 0.05 | ❌ |
| P95 latency | 7.71 s | < 8 s | ✅ |

Informational: outcome accuracy 15/15, archetype accuracy 15/15, HITL
escalation 15/15, 80,929 tokens for the whole run (~5,400 per query).

### Findings

- **The calibration error is mostly a property of the dataset.**
  - All 15 cases are answered correctly, while the mean confidence is 0.89,
    so the 0.11 gap is under-confidence on easy cases.
  - The dataset has no case where the agent is wrong, so ECE cannot reward
    good calibration yet. The next step is to add failure cases, such as a net
    loss or a top-scoring account that should not win, and then recalibrate
    the 0.70 threshold.
- **Latency is dominated by the LLM calls.**
  - Mean per-node times: profiling 2.2 s, advisory 2.3 s, critic 0.9 s.
  - Retrieval (0.1 s) and the whole deterministic evaluation (< 0.02 s) are
    negligible.
- **A critic false positive was the main cause of the tail latency.**
  - The previous run had P95 8.56 s. The LLM critic kept rejecting the
    sentence about an account that pays more at a higher balance: the writer
    was told to produce it, but the critic could not see why it was
    supported.
  - Each rejection cost a revision round, and the rewrite usually fell back
    to the template for lack of budget.
  - That sentence is now always built deterministically from the calculator
    figures and kept out of the critic's input. P95 dropped to 7.71 s.
  - Tokens went up (64.8k → 80.9k), because more answers now keep the LLM
    narrative and its critique instead of ending on the template.

## Limitations

- **Catalog coverage.**
  - 118 products from public CASFIM / CONDUSEF data.
  - Several products have no RECA document of their own, so only their
    catalog figures are checked.
  - A few yields are estimates: a share of CETES, or gap fills between
    official tiers. They are flagged as estimates in the answer.
- **Point-in-time data.** Rates and fees change. Evidence older than 30 days
  makes the agent abstain or escalate, but the catalog itself has to be
  rebuilt to stay current.
- **Synthetic benchmark.** 15 cases cover the archetypes and the guardrail
  triggers, but they are not real user traffic.
- **Language.** The interface and answers are in English. The source
  documents are in Spanish, and the multilingual retrieval handles the gap.
- **Not financial advice.** Every answer carries the legal disclaimer, and
  high-amount cases go to a human.

## License

MIT, see [LICENSE](LICENSE).
