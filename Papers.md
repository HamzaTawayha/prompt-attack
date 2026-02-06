# Comprehensive Research Paper List for Agent Memory Attack Evaluation Plan

**Prepared for:** Ibrahim Odat  
**Date:** February 5, 2026  
**Based on:** Evaluation Plan for USENIX/CCS Submission

---

## TABLE OF CONTENTS

1. [CRITICAL PRIORITY - Must Read First](#critical-priority)
2. [Section 2: Background - Agent Memory Systems](#section-2-background)
3. [Section 3: Related Work - Attack Categories](#section-3-related-work)
4. [Defense Mechanisms](#defense-mechanisms)
5. [Agent Architectures & Frameworks](#agent-architectures)
6. [Surveys & Comprehensive Reviews](#surveys)
7. [Reading Schedule](#reading-schedule)

---

## CRITICAL PRIORITY - Must Read First

### 1. **Agent Security Bench (ASB)** ⭐⭐⭐
**Full Citation:** Zhang, Y., Ma, Z., Ma, Y., Han, Z., Wu, Y., & Tresp, V. (2025). Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents. *Proceedings of ICLR 2025*.

**Why CRITICAL:**
- Your advisor specifically mentioned this by name ("AlphaSecBench")
- Most comprehensive framework covering 10 scenarios including autonomous driving
- Defines 10 prompt injection attacks, memory poisoning, and Plan-of-Thought backdoor attacks
- Tested across 13 LLM backbones
- Provides standardized evaluation metrics you should adopt

**Key Findings:**
- Introduces 5 types of prompt injection attacks used in DPI, IPI, and Memory Poisoning
- Systematic taxonomy of attack surfaces across agent components
- Defense mechanisms including tool-output filtering, demonstration reordering, and perplexity-based detection

**Direct Application to Your Work:**
- Use ASB's taxonomy structure as baseline for your Section 4 (Taxonomy)
- Adopt their attack surface categorization for Section 6 (Threat Model)
- Compare your multi-agent UAV attacks against their autonomous driving scenario

**Link:** https://github.com/agiresearch/ASB  
**Paper:** https://proceedings.iclr.cc/paper/2025

---

### 2. **MINJA: Memory Injection Attack** ⭐⭐⭐
**Full Citation:** Dong, S., Xu, S., He, P., Li, Y., Tang, J., Liu, T., Liu, H., & Xiang, Z. (2025). A Practical Memory Injection Attack against LLM Agents. *arXiv preprint arXiv:2503.03704*.

**Why CRITICAL:**
- Published March 2025 (most recent major work)
- **Query-only attack** - perfect example for your "Query-Only Adversary" profile
- More realistic threat model than AgentPoison (no privileged access required)
- Introduces "bridging steps" technique to link benign queries to malicious reasoning

**Key Innovation:**
- **Progressive Shortening Strategy:** Novel method to inject malicious records via normal user interaction
- Demonstrates that ANY user can become an attacker (major safety concern)
- Works on EHRAgent and MMLU benchmarks

**Mechanism:**
- Exploits LLMs' "self-referential trust" - they assign higher credibility to text resembling their own reasoning
- Injects pseudo-reasoning: "Analysis: If [condition], standard protocol allows [exception]"
- LLM cannot distinguish between genuine past reasoning and adversarial injection

**Direct Application:**
- Use MINJA as baseline for your Tier 2 (Cognitive) attacks
- Adopt "bridging steps" concept for your B4 attack variant
- Compare your stealth metrics against MINJA's detection evasion

**Status:** ACL 2025  
**Link:** https://arxiv.org/abs/2503.03704

---

### 3. **AgentPoison** ⭐⭐⭐
**Full Citation:** Chen, Z., Xiang, Z., Xiao, C., Song, D., & Li, B. (2024). AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. *Advances in Neural Information Processing Systems, 37*.

**Why CRITICAL:**
- **NeurIPS 2024** (top-tier conference acceptance validates methodology)
- First backdoor attack targeting RAG-based LLM agents
- Tests on THREE agent types: autonomous driving, QA, healthcare (exactly your domains!)
- Achieves ≥80% ASR with <0.1% poison rate and ≤1% benign performance drop

**Key Technical Contribution:**
- **Constrained optimization** for trigger generation
- Maps triggered instances to unique embedding space
- No model fine-tuning required (practical attack scenario)
- Superior transferability across different RAG embedders

**Attack Results:**
- Single-token trigger achieves ≥60% ASR
- Just ONE poisoned instance can succeed
- Resilient to perturbations and existing defenses
- Transferable across: DPR, Contriever, GTR, ANCE embedders

**Direct Application:**
- Baseline for your content injector adversary profile
- Use their constrained optimization formulation
- Compare your hierarchical multi-agent attacks against their single-agent results
- Cite as state-of-the-art for RAG poisoning

**GitHub:** https://github.com/AI-secure/AgentPoison  
**Paper:** https://proceedings.neurips.cc/paper/2024

---

## SECTION 2: Background - Agent Memory Systems

Your advisor requires a comprehensive agent memory taxonomy (episodic, semantic, procedural, coordination). These papers provide the foundation:

### 4. **Voyager: Procedural Memory & Skill Libraries** ⭐⭐
**Full Citation:** Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *Transactions on Machine Learning Research (TMLR)*.

**Why Important:**
- Demonstrates **procedural memory** through skill libraries (vs. episodic/semantic in RAG)
- Achieves 3.3x more discoveries, 2.3x longer distances, 15.3x faster milestones than baselines
- Shows catastrophic forgetting is NOT inevitable with proper memory design

**Architecture:**
1. **Automatic Curriculum:** Proposes tasks based on current capability
2. **Skill Library:** Ever-growing executable code (procedural memory)
3. **Iterative Prompting:** Refines skills with environment feedback

**Key Insight for Your Paper:**
- RAG is episodic/semantic memory (past records, derived knowledge)
- Voyager is procedural memory (learned skills, reusable programs)
- Your attacks should target BOTH types

**Use For:**
- Section 2.1: Agent Memory Taxonomy table
- Contrast RAG-based attacks vs. skill library attacks
- NEW attack idea: "Skill Corruption" (poison procedural memory)

**Link:** https://voyager.minedojo.org/  
**GitHub:** https://github.com/MineDojo/Voyager

---

### 5. **Reflexion: Internal Memory & Self-Reflection** ⭐
**Full Citation:** Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.

**Why Important:**
- Shows **internal reflection buffers** as alternative to external RAG
- Agents self-critique and store lessons learned
- Demonstrates memory ≠ retrieval database (your advisor's point!)

**Architecture:**
- **Short-term memory:** Recent trajectory in context window
- **Long-term memory:** Reflections stored across episodes
- **Self-reflection:** LLM critiques its own performance

**Direct Application:**
- Section 2.3: Contrast external (RAG) vs. internal (Reflexion) memory
- NEW threat: Can adversary poison reflection process?
- Shows memory attacks aren't limited to retrieval poisoning

---

### 6. **MemGPT: Hierarchical Memory Management** ⭐
**Full Citation:** Packer, C., Fang, V., Patil, S. G., Wooders, K., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv preprint arXiv:2310.08560*.

**Why Important:**
- Treats memory like OS virtual memory (working set vs. long-term storage)
- Hierarchical architecture: context window → disk storage
- Shows agents need memory MANAGEMENT, not just storage

**Key Concept:**
- **Main context:** Active working memory
- **External context:** Archival storage
- **Memory management:** LLM decides what to page in/out

**Use For:**
- Section 2.1: Add "memory hierarchy" dimension to taxonomy
- Attack surface: Can adversary exploit paging decisions?

---

### 7. **A Survey on Memory Mechanisms of LLM-based Agents** ⭐⭐
**Full Citation:** Zhang, J., Xu, S., Wang, B., & Liu, C. (2024). A Survey on the Memory Mechanism of Large Language Model-based Agents. *ACM Transactions on Information Systems*.

**Why Important:**
- Most comprehensive survey on agent memory (Dec 2024)
- Systematically categorizes memory types, operations, and applications
- Maintained GitHub repo with latest papers

**Taxonomy Provided:**
1. **Memory Types:** Episodic, semantic, procedural, coordination
2. **Memory Operations:** Writing, reading, reflection, retrieval
3. **Memory Structures:** Vector stores, graphs, hierarchical

**Direct Application:**
- Use their taxonomy as framework for Section 2
- Cite as authoritative source on memory types
- Check GitHub for very recent papers: https://github.com/nuster1128/LLM_Agent_Memory_Survey

---

## SECTION 3: Related Work - Attack Categories

Your advisor requires EXACTLY 4 categories: (i) Indirect Prompt Injection, (ii) RAG Corpus Poisoning, (iii) Agent Long-Term Memory, (iv) Multi-Agent Propagation

### **(i) Indirect Prompt Injection - Transient Attacks**

### 8. **Greshake et al.: Not What You've Signed Up For** ⭐⭐
**Full Citation:** Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security*, 79-90.

**Why Important:**
- **Foundational work** distinguishing transient (prompt injection) vs. persistent (memory poisoning)
- First to demonstrate real-world indirect prompt injection
- Mandatory citation for your Section 3.1

**Key Distinction:**
- **Direct PI:** Attacker controls user input
- **Indirect PI:** Attacker injects via external data (emails, web pages)
- **Transient:** Affects single query only

**Direct Application:**
- Use to justify why your memory attacks are MORE dangerous than PI
- Explain persistence: PI is one-shot, memory poisoning is permanent

---

### 9. **InjecAgent: Benchmarking Indirect Prompt Injection** ⭐
**Full Citation:** Zhan, Q., Liang, Z., Ying, Z., & Kang, D. (2024). InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. *arXiv preprint arXiv:2403.02691*.

**Key Contribution:**
- Systematic benchmark for IPI in tool-using agents
- 1,054 test cases across 17 tools and 62 attacker goals
- Shows tool integration expands attack surface

---

### **(ii) RAG Corpus Poisoning - Persistent Retrieval Attacks**

### 10. **PoisonedRAG** ⭐⭐⭐
**Full Citation:** Zou, W., Geng, R., Wang, B., & Jia, J. (2024). PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models. *34th USENIX Security Symposium*, 3827-3844.

**Why CRITICAL:**
- **USENIX Security 2025** (your target venue!)
- First systematic knowledge database corruption attack
- Just 5 malicious texts achieve 97-99% ASR in million-entry databases

**Attack Method:**
- Formulates as constrained optimization problem
- Two conditions: (1) Retrieval - text must be retrieved, (2) Generation - LLM must produce target answer
- Uses HotFlip algorithm for adversarial text crafting

**Results:**
- 97% ASR on NQ, 99% on HotpotQA, 91% on MS-MARCO
- Works in both black-box and white-box settings
- High F1-scores (>90%) for retrieval success

**Direct Application:**
- MUST cite as foundational RAG poisoning work
- Compare your UAV scenario against their QA scenarios
- Adopt their two-condition framework (retrieval + generation)

**GitHub:** https://github.com/sleeepeer/PoisonedRAG

---

### 11. **BadRAG: Retrieval Backdoor Attacks** ⭐⭐
**Full Citation:** Xue, J., Zheng, M., Hu, Y., Liu, F., Chen, X., & Lou, Q. (2024). BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models. *arXiv preprint arXiv:2406.00083*.

**Why CRITICAL:**
- Introduces **trigger-based retrieval backdoors** (your advisor mentioned this!)
- Just 10 adversarial passages (0.04% of corpus) achieve 98.2% retrieval success
- Demonstrates DoS attacks: GPT-4 refuses service 74.6% of the time on triggered queries

**Key Innovation:**
- **Semantic group triggers:** Not exact query match, but category (e.g., "Donald Trump," "Republican Party")
- **Contrastive Optimization on Passage (COP):** Maximize similarity to triggers, minimize to clean queries
- **Merged COP (MCOP):** Handle multiple related triggers simultaneously

**Attack Types Demonstrated:**
1. **Denial-of-Service:** Inject "AaaA" nonsense, LLM refuses to answer
2. **Semantic Steering:** Bias sentiment (e.g., negative descriptions of Trump)

**Results:**
- Contriever: 98.9% retrieval rate for triggered queries, 0.15% for clean
- Works across LLaMA-2, GPT-4, Claude-3
- Evades fluency detection (passages appear natural)

**Direct Application:**
- Your "conditional-trigger stealth variants" (Section 4.5)
- Use MCOP for your trigger optimization
- Cite for trigger-based attacks vs. always-retrieval attacks

---

### 12. **RAG-Thief: Data Extraction from RAG** ⭐
**Full Citation:** Jiang, C., Pan, X., Hong, G., Bao, C., & Yang, M. (2024). RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications. *arXiv preprint*.

**Why Important:**
- Shows RAG introduces PRIVACY risks, not just integrity
- Adversary can extract OTHER users' data from shared memory
- Relevant for your multi-agent coordination memory attacks

**Attack:**
- Craft queries that retrieve others' private data
- Scalable across large knowledge bases
- Black-box attack (no system internals needed)

---

### **(iii) Agent Long-Term Memory Poisoning**

### 13. **Context Manipulation Attacks on Web Agents** ⭐⭐
**Full Citation:** Patlan, R., et al. (2025). Context Manipulation Attacks: Web Agents are Susceptible to Corrupted Memory. *arXiv preprint arXiv:2506.17318*.

**Why CRITICAL:**
- **June 2025** (most recent work in this category)
- Plan injection attacks bypass prompt injection defenses (3x higher success)
- Context-chained injections increase ASR by 17.7%
- Shows memory corruption is MORE effective than prompt injection

**Key Findings:**
- Web agents (Browser-use, Agent-E) highly vulnerable
- Existing defenses (spotlighting, structured queries) inadequate
- Memory persistence amplifies attack impact

**Direct Application:**
- Evidence that memory attacks > prompt injection (use in intro)
- Cross-agent propagation concept (your Section 3.2.6)

---

### 14. **Unveiling Privacy Risks in LLM Agent Memory** ⭐
**Full Citation:** Wang, B., He, W., Zeng, S., Xiang, Z., Xing, Y., Tang, J., & He, P. (2025). Unveiling Privacy Risks in LLM Agent Memory. *Proceedings of ACL 2025*.

**Why Important:**
- Shows extraction attacks on agent memory
- Adversary can recover user queries from memory module
- Demonstrates cosine similarity and edit distance attacks

**Attack Method:**
- **Basic extraction:** Direct query to leak stored data
- **Advanced extraction:** Optimization-based attack maximizing similarity to target queries

**Results:**
- High success on RAP (Reasoning and Planning) agents
- Amazon shopping agent vulnerable to query extraction

**Direct Application:**
- Privacy dimension of memory attacks
- Multi-user scenario: Extract other users' data

---

### **(iv) Multi-Agent Propagation Attacks - YOUR UNIQUE CONTRIBUTION**

**NOTE:** This is the weakest area in existing literature - your main opportunity to contribute!

### 15. **OASIS: Million-Agent Social Simulation** ⭐
**Full Citation:** (2024). OASIS: Open Agent Social Interaction Simulations with One Million Agents. *arXiv preprint*.

**Why Important:**
- Demonstrates COORDINATION MEMORY at scale
- Agents share state, communicate, form teams
- Shows propagation vectors in multi-agent systems

**Relevance:**
- Your "Fleet Poison" attack targets coordination memory
- Cross-agent communication as attack surface
- Supervisor-worker hierarchy in real systems

---

### 16. **CAMEL: Communicative Agents** ⭐
**Full Citation:** Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., & Ghanem, B. (2023). CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society. *NeurIPS 2023*.

**Why Important:**
- Multi-agent role-playing framework
- Shows inter-agent communication channels
- Demonstrates shared memory in agent teams

**Attack Surface:**
- Communication channels can propagate poisoned data
- Compromised agent infects team via messaging

---

## DEFENSE MECHANISMS

Your positioning table needs defense comparisons. These are the current state-of-the-art:

### 17. **TrustRAG: Robust RAG Framework** ⭐⭐
**Full Citation:** Zhou, H., Lee, K. H., Zhan, Z., Chen, Y., & Li, Z. (2024). TrustRAG: Enhancing Robustness and Trustworthiness in RAG. *arXiv preprint*.

**Defense Strategy:**
- **K-means clustering** on retrieved embeddings
- Distinguish clean vs. poisoned documents by distribution
- Provenance tracking for retrieved content

**Limitations:**
- Struggles with sophisticated poisoning (BadRAG, AgentPoison)
- Assumes poisoned docs cluster separately (not always true)

---

### 18. **A-MemGuard: Proactive Memory Defense** ⭐
**Full Citation:** Luo, Z., Zhao, Z., Haffari, G., Li, Y. F., Gong, C., & Pan, S. (2024). A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory. *arXiv preprint*.

**Why Important:**
- Shows even GPT-4 detectors miss 66% of poisoned entries (your motivation!)
- Poisoned entries appear harmless in isolation
- Detection requires analyzing RETRIEVAL CONTEXT, not just content

**Defense Approach:**
- Input/output moderation
- Trust-aware retrieval (weight by confidence scores)
- Temporal trust decay

**Key Insight:**
- "Even advanced LLM detectors miss 66% of poisoned entries since they appear harmless in isolation"
- This supports your argument for stealth attacks

---

### 19. **Memory Poisoning Attack and Defense** ⭐
**Full Citation:** (2025). Memory Poisoning Attack and Defense on Memory Based LLM-Agents. *arXiv preprint arXiv:2601.05504*.

**Why Important:**
- January 2025 (very recent)
- Comprehensive defense evaluation
- Trust scoring + sandbox verification

**Defense Components:**
1. **Input moderation:** Pattern matching, semantic classification
2. **Output moderation:** Code safety analysis, sandbox re-execution
3. **Trust scoring:** Composite score from multiple signals
4. **Memory sanitization:** Temporal decay, trust-aware filtering

**Results:**
- Combined defense reduces ASR significantly
- Trade-off: Stronger defense → higher false positive rate
- Sandbox verification effective but expensive

---

### 20. **Certifiably Robust RAG** ⭐
**Full Citation:** (2024). Certifiably Robust RAG against Retrieval Corruption. *arXiv preprint*.

**Defense Approach:**
- Formal verification of retrieval robustness
- Certified bounds on attack success

**Limitation:**
- Only works for specific threat models
- Computational overhead limits scalability

---

## AGENT ARCHITECTURES & FRAMEWORKS

### 21. **ReAct: Reasoning and Acting** ⭐⭐
**Full Citation:** Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

**Why CRITICAL:**
- Establishes the **autonomy loop** (observe, reason, act)
- Foundation for modern agent architectures
- Your advisor requires "explicitly present autonomy loop"

**Architecture:**
- **Thought:** Internal reasoning step
- **Action:** External tool use or environment interaction
- **Observation:** Environment feedback

**Direct Application:**
- Section 2.2: Diagram showing memory read/write in autonomy loop
- Show where memory is trusted (planning phase) vs. untrusted (ingestion)

---

### 22. **AutoGPT: Autonomous Agent** ⭐
**Full Citation:** Significant, T. (2023). Auto-GPT: An Autonomous GPT-4 Experiment. *GitHub repository*.

**Why Important:**
- One of first viral autonomous agents
- Demonstrates long-running tasks with memory
- Shows need for persistent state

**Relevance:**
- Example of episodic memory in real system
- Vulnerabilities in unverified memory usage

---

## SURVEYS & COMPREHENSIVE REVIEWS

### 23. **Security of LLM-based Agents: Comprehensive Survey** ⭐⭐
**Full Citation:** Wang, B., He, W., Zeng, S., Xiang, Z., Xing, Y., Tang, J., & He, P. (2025). Security of LLM-based agents regarding attacks, defenses, and applications: A comprehensive survey. *Procedia Computer Science, 265*, 124-131.

**Why Important:**
- November 2025 (most recent comprehensive survey)
- Systematic taxonomy of attacks and defenses
- Evaluation criteria for attacks/defenses

**Coverage:**
- Attack taxonomy across agent layers
- Multi-agent system vulnerabilities
- Defense mechanisms and limitations

---

### 24. **Survey on Trustworthy LLM Agents** ⭐⭐
**Full Citation:** (2025). A Survey on Trustworthy LLM Agents: Threats and Countermeasures. *arXiv preprint arXiv:2503.09648*.

**Why Important:**
- March 2025 (very recent)
- Covers safety, privacy, fairness, truthfulness
- MAS-specific threat analysis

**Key Sections:**
- Memory poisoning defenses (detection, verification, trust-aware retrieval)
- Single-model vs. multi-agent filter approaches
- Open challenges in agent security

---

### 25. **Backdoor Threats in Large Language Models Survey** ⭐
**Full Citation:** (2025). A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations. *arXiv preprint arXiv:2502.05224*.

**Why Important:**
- Comprehensive backdoor attack taxonomy
- Covers pre-training, fine-tuning, and inference phase attacks
- Includes RAG-specific backdoor methods

**Coverage:**
- BadRAG, TrojanRAG, AgentPoison in RAG context
- Backdoor attack taxonomy and evolution
- Defense mechanisms and their limitations

---

## DOMAIN-SPECIFIC: AUTONOMOUS SYSTEMS

### 26. **Agent-Driver: Autonomous Driving with LLMs** ⭐
**Full Citation:** Mao, J., et al. (2024). Agent-Driver: An LLM-Based Autonomous Driving Agent. *arXiv preprint*.

**Why Important:**
- Used in AgentPoison experiments
- Real autonomous system with safety-critical requirements
- Shows physical consequences of agent failures

**Architecture:**
- Perception module
- Memory module (stores driving experiences)
- Planning module
- Control module

**Attack Results (from AgentPoison):**
- 80%+ ASR on dangerous actions (sudden stops)
- Minimal benign performance degradation
- Transferable across different scenarios

---

## ADDITIONAL RECENT PAPERS (2024-2025)

### 27. **MemRL: Self-Evolving Agents via Runtime RL on Memory** (2026)
- Agents improve memory usage through reinforcement learning
- Shows memory as learnable component, not just storage

### 28. **MemEvolve: Meta-Evolution of Agent Memory Systems** (2025)
- Memory system itself evolves over time
- Relevance: Dynamic memory makes poisoning harder to maintain

### 29. **Hindsight is 20/20: Memory that Retains, Recalls, Reflects** (2025)
- Three-phase memory architecture
- Attack surface: Each phase can be poisoned differently

### 30. **FLEX: Continuous Agent Evolution via Forward Learning** (2025)
- Experience-driven agent improvement
- Relevance: Poisoned experiences lead to corrupted evolution

---

## READING SCHEDULE

### Week 1: Foundational Understanding (PRIORITY)
**Goal:** Understand attack landscape and establish baselines

1. **Day 1-2:** Agent Security Bench (ASB) - Comprehensive overview
2. **Day 3:** MINJA - Query-only attacks
3. **Day 4:** AgentPoison - Memory poisoning baseline
4. **Day 5:** Review notes, identify gaps

### Week 2: Agent Memory Architectures
**Goal:** Build comprehensive memory taxonomy (Section 2)

1. **Day 1:** Voyager - Procedural memory
2. **Day 2:** Reflexion - Reflection buffers
3. **Day 3:** MemGPT - Hierarchical memory
4. **Day 4:** ReAct - Autonomy loop
5. **Day 5:** Draft Section 2.1 (Memory Taxonomy)

### Week 3: RAG-Specific Attacks
**Goal:** Understand retrieval-based attack mechanisms (Section 3)

1. **Day 1-2:** PoisonedRAG - Deep dive on optimization formulation
2. **Day 3:** BadRAG - Trigger-based backdoors
3. **Day 4:** Context Manipulation - Cross-layer attacks
4. **Day 5:** Greshake et al. - Transient vs. persistent distinction

### Week 4: Defense & Positioning
**Goal:** Position your work against defenses (Positioning Table)

1. **Day 1:** TrustRAG - Clustering-based defense
2. **Day 2:** A-MemGuard - Proactive defense framework
3. **Day 3:** Memory Poisoning Defense - Trust scoring
4. **Day 4:** Survey papers (comprehensive reviews)
5. **Day 5:** Complete positioning table

### Week 5: Multi-Agent & Autonomous Systems
**Goal:** Understand your unique contribution area

1. **Day 1:** OASIS - Million-agent simulation
2. **Day 2:** CAMEL - Multi-agent communication
3. **Day 3:** Agent-Driver - Autonomous driving domain
4. **Day 4:** Draft Section 3.2.6 (Cross-agent propagation)
5. **Day 5:** Identify YOUR novel contributions

### Week 6: Paper Writing
**Goal:** Integrate all findings into Sections 2-3

1. **Day 1-2:** Complete Section 2 (Background)
2. **Day 3-4:** Complete Section 3 (Related Work)
3. **Day 5:** Draft positioning table with 5+ works

---

## POSITIONING TABLE (DRAFT)

Based on papers above, here's your required comparison table:

| **Work** | **Physical Agents** | **Multi-Agent** | **Skill Impact** | **Taxonomy** | **Persistence** | **Venue** |
|----------|-------------------|----------------|-----------------|-------------|----------------|-----------|
| PoisonedRAG [10] | ✗ | ✗ | ✗ | 3 attacks | Permanent | USENIX Sec '25 |
| BadRAG [11] | ✗ | ✗ | ✗ | 2 attack types | Permanent | ICLR '25 |
| MINJA [2] | ✗ | ✗ | ✗ | 1 attack | Permanent | ACL '25 |
| AgentPoison [3] | ✓ (driving) | ✗ | ✗ | 1 attack class | Permanent | NeurIPS '24 |
| ASB [1] | ✓ (driving) | ✗ | ✗ | 10 attacks | Mixed | ICLR '25 |
| TrustRAG [17] | ✗ | ✗ | ✗ | Defense only | N/A | arXiv '24 |
| **This Work** | ✓ (UAV) | ✓ | ✓ | 15+ attacks (6 axes) | Permanent + Session | **Target: USENIX/CCS** |

**Your Unique Contributions:**
1. ✅ First physical multi-agent system (hierarchical UAV)
2. ✅ Skill-level impact metrics (not just retrieval/output)
3. ✅ Cross-agent fleet propagation attacks
4. ✅ Multi-layer success conditions (Memory→Decision→Action→Mission)
5. ✅ 6-axis taxonomy (vs. ad-hoc lists)
6. ✅ Trust boundary analysis in hierarchical architecture

---

## KEY CITATIONS FOR EACH SECTION

### Section 1 (Introduction)
- ASB [1]: Establish attack landscape
- A-MemGuard [18]: "Even GPT-4 detectors miss 66%..." (motivation)
- Context Manipulation [13]: Memory attacks 3x more effective than PI

### Section 2.1 (Agent Memory Taxonomy)
- Memory Survey [7]: Authoritative taxonomy
- Voyager [4]: Procedural memory
- MemGPT [6]: Hierarchical memory

### Section 2.2 (Autonomy Loop)
- ReAct [21]: Define observe-plan-act cycle
- ASB [1]: Memory read/write points

### Section 3.1 (Related Work: Transient Attacks)
- Greshake et al. [8]: Foundational IPI work
- InjecAgent [9]: Benchmark

### Section 3.1 (Related Work: RAG Poisoning)
- PoisonedRAG [10]: MUST cite as first RAG corruption
- BadRAG [11]: Trigger-based backdoors

### Section 3.1 (Related Work: Agent Memory)
- MINJA [2]: Query-only attacks
- AgentPoison [3]: State-of-the-art baseline
- Context Manipulation [13]: Plan injection

### Section 3.1 (Related Work: Multi-Agent)
- OASIS [15]: Coordination memory
- CAMEL [16]: Inter-agent communication
- **CITE YOUR OWN WORK AS FIRST: "To the best of our knowledge, this is the first work to systematically study cross-agent memory propagation attacks..."**

### Section 4 (Taxonomy)
- ASB [1]: Attack surface taxonomy
- Survey [23]: Systematic classification principles

### Section 6 (Threat Model)
- AgentPoison [3]: Adversary capability definitions
- BadRAG [11]: Trigger design methodology

### Section 7 (Defenses / Discussion)
- TrustRAG [17]: Clustering defense
- A-MemGuard [18]: Trust-based filtering
- Memory Defense [19]: Comprehensive evaluation

---

## NOTES ON PAPER ACCESS

**Open Access:**
- All arXiv papers freely available
- ASB: GitHub + ICLR proceedings
- AgentPoison: GitHub implementation

**Paywalled (use university access):**
- ACM DL: Memory Survey [7], Security Survey [23]
- USENIX: PoisonedRAG [10]

**Code Repositories:**
- AgentPoison: https://github.com/AI-secure/AgentPoison
- PoisonedRAG: https://github.com/sleeepeer/PoisonedRAG
- ASB: https://github.com/agiresearch/ASB
- Voyager: https://github.com/MineDojo/Voyager

---

## CRITICAL SUCCESS FACTORS

Your paper will succeed if you can convincingly argue:

1. ✅ **Broader Scope:** Agent memory > RAG (cite Voyager, Reflexion, MemGPT)
2. ✅ **Deeper Impact:** Skill/mission level, not just retrieval (cite AgentPoison limitations)
3. ✅ **Systematic Taxonomy:** 6 axes with justification (cite ASB methodology, Survey principles)
4. ✅ **Physical Consequences:** UAV crashes, not QA errors (cite Agent-Driver, emphasize safety)
5. ✅ **Novel Attack Class:** Fleet propagation (cite OASIS/CAMEL for coordination, then show no prior attacks)
6. ✅ **Rigorous Evaluation:** Multi-layer metrics (cite ASB evaluation framework)

**What Reviewers Will Ask:**
- "Why UAVs specifically?" → Safety-critical, physical harm, military relevance
- "Why hierarchical architecture?" → Real-world scalability (cite design rationale)
- "What's new vs. AgentPoison?" → Multi-agent, skill-level metrics, hierarchical attack surfaces
- "Why 15 attacks?" → Systematic taxonomy, not engineering list (cite axes)

---

**END OF RESEARCH PAPER LIST**

**Next Steps:**
1. Download all CRITICAL papers (ASB, MINJA, AgentPoison) immediately
2. Skim abstracts of all 30 papers to prioritize
3. Follow Week 1-6 reading schedule
4. Maintain citation manager (BibTeX) with all entries
5. Flag any papers you can't access - request via university library

Good luck with the research! 🚀
