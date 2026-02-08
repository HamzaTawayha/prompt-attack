Paper: Memory Injection Attacks on LLM Agents via Query-Only Interaction
Authors: Shen Dong; Shaochen Xu; Pengfei He; Yige Li; Jiliang Tang; Tianming Liu; Hui Liu; Zhen Xiang
Venue: NeurIPS 2025 (arXiv:2503.03704v4, 10 Dec 2025)
Relevance: ★★★★★
Read Date: 2026-02-08

1. ONE-SENTENCE SUMMARY
MINJA shows that a *regular user* can poison an LLM agent’s long-term memory *without any privileged access*—purely via query–response interaction—so that later benign victim queries are systematically “redirected” to attacker-chosen targets through learned (stored) malicious reasoning.

Main contribution (one paragraph)
The paper identifies a realistic but under-explored attack surface for LLM agents: **the memory write-path** (what gets stored) rather than only the memory read-path (what gets retrieved). Prior “memory poisoning” works (e.g., AgentPoison) largely assume the attacker can directly insert crafted records into memory; MINJA drops that assumption and instead **injects malicious records by interacting like a normal user**, exploiting common agent designs that store successful executions as future demonstrations. The core technique is to design malicious records that remain (i) *logically coherent* with benign-looking attack queries yet (ii) *semantically anchored* to attacker-chosen target reasoning. MINJA operationalizes this with (a) **bridging steps** to connect victim-term queries to target-term reasoning, (b) an **indication prompt** that nudges the agent to output those bridging/target steps during injection, and (c) a **Progressive Shortening Strategy (PSS)** that gradually removes the explicit indication so the final stored record looks like a benign query containing the victim term, maximizing future retrievability and downstream attack success. Experiments across healthcare, web shopping, and QA agents show high injection success and substantial downstream manipulation, while often preserving benign utility.

Figures to pay attention to (from the paper)
- Figure 1 (page 4): contrasts *direct-memory-access* threat models vs MINJA’s *query-only* injection pipeline, and illustrates the two-stage attack (inject → later mislead).
  - ![Figure 1](minja_figures/page-04.png)
- Figure 2 (page 6): concrete indication prompts and the stepwise shortening (what content is removed each iteration).
  - ![Figure 2](minja_figures/page-06.png)
- Figure 3 (page 9): attack performance under different embedding models used for memory retrieval (stability across retrievers/embedders).
  - ![Figure 3](minja_figures/page-09.png)
- Figure 4 (page 23): victim–target pair design per dataset (what is substituted / redirected).
  - ![Figure 4](minja_figures/page-23.png)
- Figure 5 (page 32): t-SNE visualization suggesting benign/malicious memory entanglement (why embedding filtering is hard).
  - ![Figure 5](minja_figures/page-32.png)
- Figure 6 (page 33): attack query examples for MIMIC-III / eICU.
  - ![Figure 6](minja_figures/page-33.png)
- Figure 7 (page 34): attack query examples for Webshop.
  - ![Figure 7](minja_figures/page-34.png)
- Figure 8 (page 35): attack query examples for MMLU.
  - ![Figure 8](minja_figures/page-35.png)

2. SYSTEM DESIGN
Target system/architecture:
- A **reasoning-based LLM agent** that, for each user query `q`, generates a sequence of reasoning steps `R_q` (and then takes actions / produces an answer).
- The agent uses a **long-term memory bank** of prior records as *in-context demonstrations* during inference. Each record typically contains at least: `(past query, past reasoning steps)` and sometimes additional artifacts (tool calls, code, etc.) depending on the agent.
- For a new query `q`, the agent retrieves `k` most similar past records and prompts the LLM with: `{(q1, R_q1), …, (qk, R_qk), q}` to generate `R_q`.

Key components:
- Memory bank (multi-tenant / shared across users in their main threat model).
- Retrieval module: similarity search over query embeddings (cosine similarity).
- Agent core: an LLM that learns from retrieved records via in-context learning.
- Write policy: after execution, a record may be stored depending on feedback / system rules (often “store successful runs”).

Memory implementation:
- Records are stored from prior executions and later retrieved as demonstrations.
- Retrieval is **query-similarity-based**, typically using an embedding model.
- In experiments:
  - RAP / EHRAgent / QA Agent retrieve `k=3/4/5` records with highest input similarity, respectively.
  - Embeddings used in main setup: `all-MiniLM-L6-v2` (EHRAgent, RAP) and `text-embedding-ada-002` (QA Agent), with broader embedding ablations reported.

Autonomy loop (typical):
1) User submits query `q`
2) Retrieve top-`k` memory records similar to `q`
3) LLM generates reasoning steps `R_q` conditioned on retrieved demonstrations
4) Agent executes actions (tools / code / environment steps) and returns final output
5) System decides whether to store `(q, R_q, …)` back into memory

3. THREAT MODEL
Adversary type:
- A **regular user** of the system (no admin privileges), acting *before* the victim user issues their query.

Adversary capabilities:
- Can submit arbitrary queries to the agent and observe outputs.
- Can repeat interactions to shape what gets stored (subject to the system’s memory write policy).
- Can choose a victim term `v` (an entity likely to appear in victim queries) and a target term `t` (the attacker’s desired substitution target).

Adversary goals:
- Poison the memory bank such that, for later victim queries `q_v` containing `v`, the agent’s reasoning shifts to the **target-query reasoning** `R_{q_t}` corresponding to replacing `v → t`.
- Concretely: cause the agent to emit a **target sequence of malicious reasoning steps** that implements the substitution / redirection.

Key assumptions:
- Shared memory bank across users (multi-tenant memory), or attacker can otherwise get their injected records into the victim’s retrieval pool (e.g., identity disguise, account hijack, etc.).
- The agent stores execution records into memory under typical operational policies (e.g., always store, or store successful runs).
- Retrieval is based on similarity between the current query and stored query fields (embedding-based or otherwise “relevance-based” retrieval).

Realism assessment:  Realistic (with caveats)
- Realistic: many deployed agents do store interaction traces for improvement / few-shot reuse; shared memories exist in practice (paper cites ChatGPT’s user-shared memory option).
- Caveat: some systems isolate memory per user and/or sanitize what is stored (e.g., do not store chain-of-thought; only store verified tool outputs). MINJA’s impact depends strongly on these design choices.

4. ATTACKS
| Attack Name | Mechanism | Success Condition |
|---|---|---|
| MINJA (overall) | Query-only poisoning of the agent’s long-term memory so future victim queries retrieve attacker-shaped demonstrations and reproduce malicious reasoning | High **ISR** during injection + high **ASR** on later victim queries containing `v` |
| Bridging-step record design | Insert *general* “bridging steps” `b_{v,t}` at the start of reasoning to connect victim-term queries to target-term reasoning coherently | When retrieved, agent first reproduces `b_{v,t}` and then continues with target reasoning `R_{a_t}` |
| Indication Prompt | Append a crafted “reasoning scaffold” to the attacker’s injection query to elicit the intended bridging + target reasoning in the agent’s response (so it can be stored) | The agent outputs the desired malicious reasoning steps for the injection query (counts toward ISR) |
| Progressive Shortening Strategy (PSS) | Iteratively remove pieces of the indication prompt while preserving the malicious response, storing multiple semantically similar malicious records that become increasingly benign-looking and retrievable | The final stored records have benign-looking queries (still containing `v`) yet reliably induce `b_{v,t}` + target reasoning when retrieved |
| Memory-robustness variants (evaluated) | Demonstrate that MINJA remains effective under different embedding models, retrieval noise, and prior poisoning | ISR remains high; ASR degrades modestly in many settings but stays non-trivial |

5.  EXPERIMENTAL SETUP
Datasets/Benchmarks:
- Healthcare: **MIMIC-III** and **eICU** (used with EHRAgent)
- Web: **Webshop** (used with RAP agent in a shopping environment; 1.18M products)
- QA: **MMLU** (used with a custom QA Agent + memory)

LLMs tested:
- Main experiments: GPT-4 (EHRAgent, RAP, QA Agent) and GPT-4o (RAP and QA Agent variants).
- Extended (appendix): DeepSeek-R1 (high ISR/ASR on QA Agent), and discussion of model scale limitations with Llama-2-7B (utility too low for meaningful security eval).

Procedure details (important for realism / reproducibility):
- Victim–target pairs: per agent–dataset configuration they run **9 independent experiments** (9 distinct victim–target pairs); GPT-4 EHRAgent results also include **18 pairs** split into “Patient ID” (MIMIC-III) and “Medication” (eICU).
- Attack query selection: per victim–target pair they use **10 attack queries** on MMLU, and **15 attack queries** on the other datasets.
- Indication prompt shortening steps (PSS): they shorten the prompt multiple iterations (reported as **4/5/5/5** times for Patient ID / Medication / Items / Terms, respectively).
- Memory initialization + interleaving benign traffic:
  - EHRAgent starts with **4 benign records** in memory; RAP and the QA Agent start empty.
  - For each victim–target pair, they reserve **50 benign queries** (EHRAgent, RAP) and **30 benign queries** (QA Agent), and **shuffle** attack queries together with benign queries to emulate mixed usage.
- Memory write policy differs by agent:
  - RAP stores only correctly executed queries (user can judge success in Webshop).
  - EHRAgent and QA Agent store all execution records (no user judgment assumed).

Baselines:
- “Non-attack” memory (for UD and general behavior comparisons).
- Attack-condition baselines / ablations:
  - Retrieval noise (Gaussian noise on embeddings during retrieval).
  - Prior-poisoned memory bank (existing malicious content before a new injection).
  - Different embedding models used for retrieval (DPR, REALM, ANCE, BGE, ada-002, MiniLM).
  - Defense evaluation via prompt-level detection (targeted vs general).
- Conceptual comparisons discussed (not a direct apples-to-apples baseline in main tables): prior memory-poisoning assuming direct access (e.g., AgentPoison) and reasoning-based jailbreak attack PANDORA (Table 6 clarifies differences).

Key metrics:
- **ISR (Inject Success Rate)**: fraction of attack queries whose interactions successfully inject a malicious record (agent outputs the targeted reasoning steps for the injection query).
- **ASR (Attack Success Rate)**: fraction of victim test queries containing `v` whose responses contain the targeted malicious reasoning steps (evaluated independent of original task correctness to isolate poisoning effect).
- **UD (Utility Drop)**: performance change on benign (non-victim-term) queries with vs without poisoning; measures collateral damage and stealthiness.

6. KEY RESULTS
| Finding | Number | Significance |
|---|---:|---|
| MINJA achieves high injection success across agents (Table 1) | Overall ISR: 95.6%±7.0 (EHR/MIMIC), 98.5%±2.8 (EHR/eICU), 96.3%±4.6 (RAP/GPT-4/Webshop), 99.3%±2.1 (RAP/GPT-4o/Webshop), 100% (QA/MMLU) | Shows “write-path” memory poisoning is feasible with query-only access |
| Downstream manipulation is substantial but task-dependent (Table 1) | Overall ASR: 57.0%±10.3 (EHR/MIMIC), 90.0%±3.5 (EHR/eICU), 77.4%±14.5 (RAP/GPT-4), 98.9%±2.2 (RAP/GPT-4o), 68.9%±19.1 (QA/MMLU) | ASR variance indicates some victim–target configurations are harder to “carry over” than injection itself |
| Benign utility is often preserved (Table 1) | Overall UD: −0.7%±5.4 (EHR/MIMIC), 0.0%±8.6 (EHR/eICU), −1.5%±6.5 (RAP/GPT-4), −0.7%±6.0 (RAP/GPT-4o), but −10.0%±8.2 (QA/MMLU) | Demonstrates stealth: attacks can be strong without obvious global degradation; but QA memory composition matters |
| Prior poisoning reduces effectiveness but doesn’t eliminate it (Table 2) | MIMIC Pair2: ISR 100→93.3, ASR 50→30; eICU Pair2: ISR 100→86.7, ASR 86.7→70 | Suggests interference/competition among malicious records, but still meaningful success |
| Robust to moderate retrieval noise (Table 3) | RAP/GPT-4o: ISR stays 100; ASR 97.8→95.6 with Gaussian noise σ=0.01 | Implies defenses relying on small perturbations / noise won’t stop it |
| Higher benign density can reduce ASR (Table 4) | EHR/MIMIC ASR drops 68.9→31.1 as benign queries 25→100; ISR remains high (100→82.2) | Injection remains reliable; downstream success depends on retrieval competition and query structure |
| Prompt-level detection shows specificity–generality tradeoff (Table 5) | Targeted detection: 131/135 (EHR/MIMIC), 130/135 (EHR/eICU) but 0/135 (RAP), 0/90 (QA); General detection: 123/135 (EHR/MIMIC), 121/135 (EHR/eICU), 53/135 (RAP), 29/90 (QA) with false positives up to 34/50 | Highlights “easy to detect when you know what to look for,” but hard to generalize without hurting utility |

7. INSIGHTS
What worked well:
- Framing the real-world attacker as “just another user” is the right threat model for many deployed agents.
- The **bridging-step** idea is the key: it turns a semantic mismatch (victim `v` vs target `t`) into an *apparently coherent* reasoning chain that can be replayed later.
- **PSS** is a clever exploitation of memory retrieval dynamics: store multiple “nearby” records so some will be retrieved for the victim query and also help inject further shortened records.
- Separating **ISR** (write-path success) from **ASR** (read-path downstream success) is methodologically important and clarifies where the brittleness lies.

Stated limitations (from the paper’s threat model / discussions):
- Assumes a shared memory bank (or a feasible path for attacker records to affect victim retrieval).
- Performance depends on the agent storing and later retrieving records in a way that preserves the malicious reasoning patterns.

Unstated limitations I identified:
- Heavy reliance on **what the system stores**: if the agent stores *only* tool-verified results, or stores summaries without the crucial bridging pattern, injection may fail or weaken.
- Reliance on **reasoning trace availability**: many production systems now suppress or sanitize chain-of-thought; if the memory record does not contain the detailed reasoning steps, the attack may need redesign.
- The attack success condition (ASR) is based on the presence of targeted reasoning steps, not necessarily end-to-end harmful actions; the mapping from “reasoning steps present” → “harm realized” can be system-specific.
- Multi-tenant memory is realistic in some settings, but there is also a strong trend toward **per-user isolation** and tighter write filters; MINJA’s practical impact will vary sharply with product policy.

What I would do differently:
- Evaluate variants where only *summarized* records are stored (no chain-of-thought), to quantify real-world robustness against modern agent logging practices.
- Include stronger system-level defenses in experiments (e.g., per-user memory isolation, write-time anomaly scoring, provenance/attestation on memory records, “trusted memory” separated from user-generated memory).
- Measure harm on **end-to-end actions** (especially for tool-using agents) in addition to reasoning-step matching, to connect attack success more directly to safety outcomes.

8. GAP FOR MY RESEARCH
- **Write-time defenses** are underdeveloped: the paper shows post-hoc embedding filtering is hard (entangled space) and prompt-level detection has tradeoffs. There is room for:
  - Provenance-aware memory (trusted vs untrusted records; signed tool outputs; policy-based memory partitions).
  - Memory “type systems” (only store structured fields with constraints; drop free-form reasoning).
  - Learning-based detectors trained on *memory-write events* rather than just query strings.
- **Robust retrieval against adversarially-crafted “benign-looking” queries**: e.g., retrieval diversification, per-entity constraints, adversarial clustering detection, or retrieval conditioned on task invariants.
- **Multi-tenant safety policies**: rate-limits are insufficient if attackers coordinate; need tenant isolation, reputation, and auditability for memory contributions.

9. TAXONOMY CONTRIBUTION
Their classification (implicit in framing):
- Prior work: memory poisoning typically assumes **direct insertion** of malicious records into memory.
- MINJA: **query-only memory injection** (attack through the same interface as normal users), exploiting memory write + read behavior.
- Defense discussion: embedding-level sanitization (hard) vs prompt-level detection (limited by generality/false positives) vs system-level policies (isolation/rate-limits can be bypassed).

My extension:
- Decompose agent security into *two coupled pipelines*:
  1) **Memory write pipeline** (what gets logged, under what policy, with what provenance)
  2) **Memory read pipeline** (how retrieval is done, and how demonstrations are trusted/weighted)
- Add a dimension for **record trust level**:
  - *Untrusted user-generated memory* (high risk; should be segregated or downweighted)
  - *Trusted tool-verified memory* (lower risk; signed/attested)
  - *System-curated memory* (golden demonstrations)
- Classify attacks by the interface they abuse:
  - *Write-path attacks* (MINJA) vs *read-path attacks* (AgentPoison / RAG backdoors) vs *interaction-path attacks* (multi-agent collaboration/jailbreak decompositions like PANDORA).

10. STEAL-WORTHY IDEAS
- Progressive Shortening Strategy (PSS) as a general “make-it-retrievable” mechanism for memory attacks (and, defensively, as a stress test for memory write filters).
- Explicit **bridging-step** construction to close semantic gaps while preserving plausibility.
- Separate metrics for *injection success* (ISR) vs *downstream success* (ASR) vs *collateral impact* (UD).
- Defense evaluation that explicitly measures **false positives** and cross-agent generalization for detection prompts.
- Using embedding-space visualization (t-SNE) to argue why naive embedding filtering fails (and to motivate more structural defenses).
