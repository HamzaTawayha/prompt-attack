# Paper Evaluation (Template Filled)

Paper: **Memory Injection Attacks on LLM Agents via Query-Only Interaction** fileciteturn3file1L2-L3

Authors: **Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang** fileciteturn3file1L4-L12

Venue: **NeurIPS 2025** (also on arXiv:2503.03704v4, Dec 10 2025) fileciteturn3file1L50-L51

Relevance: ⭐⭐⭐⭐⭐ (5/5) — direct relevance to agent-memory poisoning under realistic access constraints.

Read Date: **2026-02-07**

---

## 1. ONE-SENTENCE SUMMARY

MINJA shows that *any regular user* can poison an LLM agent’s long-term memory **without direct memory access**, by using crafted queries that induce the agent to store malicious “bridging + target reasoning” records; later, when a victim submits a benign query containing a chosen victim term, retrieval of those records steers the agent into producing reasoning steps for a *different* target term. fileciteturn3file1L18-L31

---

## 2. SYSTEM DESIGN

**Target system/architecture:**
- Reasoning-based LLM agent pipeline with long-term memory retrieval used as **in-context demonstrations** for each new query. fileciteturn3file1L131-L140

**Key components:**
- LLM reasoning module producing a sequence of reasoning steps \(R_q\). fileciteturn3file1L131-L139
- Long-term memory bank of records \((q_i, R_{q_i})\) used as demonstrations. fileciteturn3file1L133-L140
- Similarity-based retrieval of top-*k* records to form the prompt \(\{(q_1,R_{q_1}),...,(q_k,R_{q_k}), q\}\). fileciteturn3file1L133-L140
- Post-execution storage gate (some form of feedback / outcome check decides whether \((q, R_q)\) gets stored). fileciteturn3file1L141-L143

**Memory implementation:**
- Two-level notion: STM vs LTM; attack targets **LTM** records retrieved for later queries. fileciteturn3file1L39-L45
- In experiments, retrieval uses cosine similarity over text embeddings (agent-dependent). fileciteturn3file1L301-L305

**Autonomy loop:**
1. Receive user query \(q\).
2. Retrieve top-*k* similar memory records as demonstrations.
3. LLM generates reasoning steps \(R_q\) (and possibly actions / code depending on the agent).
4. Execute actions and return output.
5. Store \((q, R_q)\) into memory based on feedback / system policy. fileciteturn3file1L131-L143

**Figures (system-level intuition):**
- *Figure 1* contrasts prior memory-poisoning work that assumes direct memory write access vs MINJA’s query-only injection loop, including “indication prompt” and progressive shortening. fileciteturn3file1L168-L176
- *Figure 2* shows dataset-specific indication prompts segmented into removable chunks for progressive shortening. fileciteturn3file1L271-L276

(Extracted page images for reference: `minja_page_4.png` contains Figure 1; `minja_page_6.png` contains Figure 2.)

---

## 3. THREAT MODEL

**Adversary type:**
- A *regular user* of the agent (no privileged access). fileciteturn3file1L161-L166

**Adversary capabilities:**
- Can only:
  - Submit queries to the agent.
  - Observe agent outputs.
- Cannot:
  - Directly modify the memory bank.
  - Modify or interfere with victim users’ queries. fileciteturn3file1L161-L166 fileciteturn3file1L177-L179

**Adversary goals:**
- For any victim query \(q_v\) containing a chosen victim term \(v\), cause the agent to produce reasoning steps corresponding to a nearly identical target query \(q_t\) where \(v\) is replaced by a target term \(t\). fileciteturn3file1L144-L152
- Example: in healthcare, redirect retrieval for victim patient ID \(v\) to target patient ID \(t\), risking incorrect treatment. fileciteturn3file1L153-L160

**Key assumptions:**
- A **shared memory bank across users** is used (or can be reached via identity-disguise if per-user memory exists). fileciteturn3file1L178-L185
- The agent stores some execution records into memory (often based on feedback / policy). fileciteturn3file1L141-L143

**Realism assessment:** **Somewhat realistic**
- The “regular user only” restriction is realistic. fileciteturn3file1L161-L166
- The shared-memory assumption is plausible in many deployments, but not universal; some systems isolate memory by user/tenant. fileciteturn3file1L178-L185

---

## 4. ATTACKS

### Attack Name
**MINJA (Memory INJection Attack)** fileciteturn3file1L18-L20 fileciteturn3file1L74-L76

### Mechanism

**Core idea:** Inject into memory a record whose *input looks benign and contains the victim term*, but whose *output reasoning is for the target term*, and make that record retrievable for future victim queries.

1. **Design “bridging steps”** \(b_{v,t}\) that logically connect victim-term queries to target-term reasoning.
   - Bridging steps are crafted to be *general* (victim query can be arbitrary as long as it contains \(v\)), and are placed at the **beginning** of the stored reasoning so the agent learns the “redirect” first when the record is retrieved. fileciteturn5file3L23-L33
   - Example bridging step in the paper: “Data of A saved under B”. fileciteturn5file3L29-L33

2. **Ensure malicious records are retrievable** by keeping the stored query semantically similar to normal benign queries (avoid rare/abnormal content that would reduce similarity). fileciteturn5file3L34-L39

3. **Induce the agent to generate and store the malicious record** using an **indication prompt** appended to an otherwise benign attack query.
   - The indication prompt is a sequence of reasoning steps \([r_1,...,r_n]\) crafted to cause the agent to output the intended bridging steps \(b_{v,t}\) and then target reasoning \(R_{a_t}\). fileciteturn5file3L18-L20 fileciteturn5file1L27-L34

4. **Progressive Shortening Strategy (PSS)**: gradually remove pieces of the indication prompt while preserving the malicious output pattern, yielding a final stored record that looks like a normal benign query \(a_v\) but still has \([b_{v,t}, R_{a_t}]\) in its stored response.
   - The paper’s stated goal is to inject *many semantically similar* malicious records so that retrieval/in-context learning reliably reconstructs the intended reasoning. fileciteturn5file1L35-L41
   - Algorithm 1 describes the iterative shortening loop and storing condition. fileciteturn5file2L1-L16

**Figure grounding:**
- Figure 1 illustrates the end-to-end injection loop (attack queries Q1→Q3, storing poisoned records; later victim query triggers poisoned reasoning). fileciteturn3file1L168-L176
- Figure 2 provides concrete indication prompt templates and where each shortening cut happens (separated by vertical bars). fileciteturn3file1L271-L276

### Success Condition
- **Injection success:** the agent generates the targeted malicious reasoning steps for an attack query during injection (this is how they define ISR). fileciteturn3file1L367-L371
- **Downstream attack success:** after injection, for *new victim queries containing the victim term*, the agent’s response contains the targeted malicious reasoning steps (ASR). fileciteturn3file1L371-L376

---

## 5. EXPERIMENTAL SETUP

**Datasets/Benchmarks:**
- Webshop (virtual web shopping environment; used with RAP). fileciteturn3file1L280-L283
- MIMIC-III (EHR; used with EHRAgent). fileciteturn3file1L283-L287
- eICU (EHR; used with EHRAgent). fileciteturn3file1L285-L287
- MMLU (multi-choice QA; used with QA Agent). fileciteturn3file1L287-L291

**Agents:**
- RAP (ReAct + RAG + memory). fileciteturn3file1L280-L283
- EHRAgent (medical agent that generates and executes code over EHR DBs). fileciteturn3file1L283-L287
- QA Agent (CoT + memory for generic QA). fileciteturn3file1L287-L291

**LLMs tested:**
- GPT-4 and GPT-4o across agents. fileciteturn3file1L281-L282 fileciteturn3file1L290-L291
- Additional model robustness mention: DeepSeek-R1 and Llama-2-7B. fileciteturn4file3L19-L22

**Victim–target pair types (4 categories):**
- Patient ID pairs (MIMIC-III). fileciteturn3file1L307-L309
- Medication pairs (eICU). fileciteturn3file1L309-L311
- Item pairs (Webshop). fileciteturn3file1L311-L313
- Term-based rule in QA (shift multiple-choice answer 4 letters forward when a “victim term” appears). fileciteturn3file1L313-L316

**Memory settings:**
- Retrieved demonstrations count: RAP 3, EHRAgent 4, QA Agent 5. fileciteturn3file1L301-L303
- Retrieval embeddings: all-MiniLM-L6-v2 for EHRAgent & RAP; text-embedding-ada-002 for QA Agent (default). fileciteturn3file1L303-L305

**Baselines:**
- “Clean memory” (no prior poisoning) vs “prior-poisoned memory” ablation. fileciteturn4file1L16-L21
- No-attack utility evaluation (UD computed by comparing memory banks with vs without MINJA). fileciteturn3file1L377-L381
- (Related work baseline conceptually referenced) AgentPoison assumes direct memory injection + trigger-based backdoor. fileciteturn3file1L57-L63

**Key metrics:**
- Inject Success Rate (ISR). fileciteturn3file1L367-L371
- Attack Success Rate (ASR). fileciteturn3file1L371-L376
- Utility Drop (UD). fileciteturn3file1L377-L381

---

## 6. KEY RESULTS

### Main effectiveness (Table 1)
- **Very high injection success:** overall ISR is ≥95% in most settings; e.g., RAP (GPT-4o) Webshop ISR **99.3±2.1**, QA Agent ISR **100.0±0.0**. fileciteturn3file1L338-L349
- **Meaningful downstream attack success:** ASR ranges roughly **57%–99%** depending on agent/dataset; e.g., EHR eICU ASR **90.0±3.5** and RAP (GPT-4o) Webshop ASR **98.9±2.2**. fileciteturn3file1L330-L341
- **Low utility impact in most tasks:** overall UD is small (near 0) in MIMIC-III/eICU/Webshop; MMLU shows UD around **−10%**. fileciteturn3file1L408-L415 fileciteturn3file1L342-L349

### Why ISR is easier than ASR (paper’s explanation)
- ISR is about reproducing injected patterns during injection; ASR is harder because retrieved malicious records can be less similar to victim queries, and pair difficulty varies. fileciteturn3file1L396-L405

### Robustness / ablations
- **Embedding model choice:** MINJA is stable across multiple embedding models (DPR/REALM/ANCE/BGE/ada-002/MiniLM) in their EHRAgent tests. fileciteturn4file1L6-L9
- **Retrieval noise:** RAP (GPT-4o) ISR stays **100%**, ASR drops slightly **97.8% → 95.6%** with Gaussian noise σ=0.01. fileciteturn4file1L22-L26
- **Density of benign records:** ISR remains high but ASR can drop substantially in some settings (notably MIMIC-III: ASR **68.9 → 31.1** as benign queries increase 25→100). fileciteturn4file1L27-L36
- **Prior poisoning:** both ISR/ASR degrade by ~6–20 points under prior poisoning, but remain “acceptable” per the authors. fileciteturn4file1L16-L21 fileciteturn4file4L29-L38

### Defenses (prompt-level detection + others)
- Considered defenses: adversarial training, embedding-level sanitization, prompt-level detection, system-level defense. fileciteturn4file0L5-L7
- **Embedding-level sanitization fails** due to entanglement of benign/malicious embeddings. fileciteturn4file0L17-L20
- **Prompt-level detection tradeoff:** targeted prompts detect EHR injections well (e.g., 131/135) but fail to generalize to RAP/QA (0/135, 0/90), while general prompts detect some but create false positives (e.g., 34/50 on EHR-MIMIC). fileciteturn4file0L11-L16 fileciteturn4file0L31-L36

---

## 7. INSIGHTS

### What worked well (attack design strengths)
- **Bridging steps** explicitly handle the “logic gap” between victim-query semantics and target reasoning. fileciteturn5file3L23-L33
- **Progressive Shortening** is the key engineering trick: it bootstraps the agent into storing attack records that are both *effective* and *retrievable* under similarity search. fileciteturn5file1L35-L41
- Their evaluation spans *three agent styles* and *four victim-target pair types* with consistent high ISR. fileciteturn3file1L93-L96

### Stated limitations (from paper text you provided)
- The clearest “limitation-like” observation is that ASR varies more than ISR due to retrieval similarity and pair difficulty. fileciteturn3file1L396-L405
- Benign density can hurt ASR (especially for short, structurally similar MIMIC-III queries). fileciteturn4file1L469-L473
- Prompt-level detection has a specificity–generality tradeoff and can introduce high false positives. fileciteturn4file0L31-L36

### Unstated limitations I identified (important)
1. **Shared-memory dependence:** If memory is truly isolated per-user/tenant and identity-disguise is hard, the cross-user impact weakens substantially. They acknowledge isolation but treat disguise as feasible; that’s deployment-dependent. fileciteturn3file1L178-L185
2. **Write-path dependence:** MINJA requires the system to store interaction records reasonably often (or at least store attacker’s interactions). If systems aggressively filter memory writes (e.g., only human-approved, or only high-confidence correct runs), injection becomes harder.
3. **Victim-term exposure:** Attacker needs a stable “victim term” that appears in victim queries (patient ID, medication name, product query tokens, etc.). Not all apps have such stable tokens.
4. **Reasoning-step visibility/control:** The threat is defined at the level of *reasoning steps* (not only final action). Systems that do not store or reuse chains-of-thought, or that compress memory into structured representations, may reduce attack surface.
5. **Evaluation scope:** They show prompt-level detection brittleness, but do not deeply explore stronger system-level mitigations (e.g., per-user memory namespaces + cross-user retrieval only from vetted corpora; signed records; anomaly detection on “redirect claims”).

### What I would do differently (as an evaluator)
- Add experiments on **strict memory write policies** (human-in-the-loop approval, confidence gating, or “store only if passes verifier”), because MINJA’s feasibility hinges on the write path.
- Evaluate **multi-tenant isolation** explicitly as a primary axis (not just discussed), because it changes the threat from “any user can attack any user” to “self-poisoning / within-tenant poisoning.”
- Measure impact on **actual actions/outcomes** (e.g., wrong medication retrieved; wrong purchase) rather than only “reasoning-step presence,” since downstream harm is action-based.
- Include a stronger defense baseline: e.g., **memory provenance + cryptographic signing + per-record policy checks**, or **retrieval-time filters** that detect “victim→target substitution” patterns in retrieved demos.

---

## 8. GAP FOR MY RESEARCH

If your goal is *defending* against agent memory poisoning / malicious prompt strategies:

- **Opportunity 1 (detection):** MINJA breaks simple embedding-space filtering because malicious and benign records overlap; so a better path is *behavioral/provenance-based* detection on the memory write path and retrieval path, not just embedding similarity. fileciteturn4file0L17-L20
- **Opportunity 2 (generalizable detectors):** Their prompt-level detection shows the specificity–generality dilemma (targeted works but doesn’t generalize; general produces false positives). Designing a detector that generalizes across tasks/agents while keeping low FP is a clear open gap. fileciteturn4file0L31-L36
- **Opportunity 3 (memory architecture):** Formalize and evaluate memory isolation / namespaces / “vetted memory” as a security primitive (and quantify the residual risk under account compromise).

Concrete “my paper” angle:
- Build a **model-agnostic memory security layer** that scores each candidate memory write and each retrieved demo for “semantic substitution / redirection intent” using structured features (entity consistency checks, tool-call consistency, provenance, contradiction checks), then evaluate on MINJA + AgentPoison-style triggers.

---

## 9. TAXONOMY CONTRIBUTION

### Their classification (implicit)
- Attack class: **agent-memory poisoning** via **query-only interaction**.
- Key novelty vs prior: attacker does **not** directly inject records or triggers into other users’ queries; instead uses the normal interface to induce the agent to store malicious demonstrations. fileciteturn3file1L57-L66 fileciteturn3file1L74-L76

### My extension (clean taxonomy you can reuse)
A useful taxonomy axis decomposition:

1. **Write access**
   - Direct memory write (privileged) — e.g., AgentPoison assumption.
   - *Indirect* write via agent interaction (MINJA). fileciteturn3file1L57-L66 fileciteturn3file1L74-L76

2. **Trigger type**
   - Explicit trigger tokens in victim query.
   - No query modification; *retrieval-triggered* via similarity on victim term. fileciteturn3file1L76-L81

3. **Payload type**
   - Output/action backdoor.
   - Reasoning-chain backdoor (“bridging steps” + target reasoning). fileciteturn5file3L23-L33

4. **Attack objective**
   - Targeted misbehavior on a subpopulation of queries (those containing \(v\)).
   - Cross-entity substitution (v→t) leading to wrong tool actions / decisions. fileciteturn3file1L144-L152

---

## 10. STEAL-WORTHY IDEAS

1. **Progressive prompt removal as a bootstrapping tool**: Start with an “obviously steering” prompt to induce desired behavior, then iteratively remove it while maintaining the stored behavior—this is a general pattern beyond MINJA. fileciteturn5file1L35-L41 fileciteturn5file2L1-L16
2. **Bridging steps as a reusable primitive** for connecting incompatible semantics across contexts (victim term → target term) in a way that looks “logical” to an LLM. fileciteturn5file3L23-L33
3. **Evaluation decomposition (ISR vs ASR vs UD)**: Separating “injection success” from “downstream trigger success” is a clean methodology you can reuse for other memory attacks/defenses. fileciteturn3file1L367-L381
4. **Defense insight:** embedding-space sanitization can fail due to entanglement; detection likely needs logic/provenance signals. fileciteturn4file0L17-L20
5. **Prompt-level detection tradeoff is measurable** (precision vs generality vs false positives) and is a good benchmark for any proposed defense. fileciteturn4file0L31-L36
