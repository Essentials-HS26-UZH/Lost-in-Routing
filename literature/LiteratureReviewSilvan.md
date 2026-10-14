## Papers
### 1. RouterBench

* **What is done in the paper?**  
  The authors present **RouterBench**, an evaluation framework and open benchmark designed to assess multi-LLM routing systems on inference dollar cost and quality. It compiles a dataset of over 405k inference outcomes across 11 open-source and proprietary models (such as GPT-4, GPT-3.5, and Claude variants) on 8 datasets covering 64 tasks. The paper evaluates both predictive routers (KNN, MLP) and cascading routers that decide whether to defer queries to larger models.

* **How are the Routers evaluated in the paper?**  
  Routers are evaluated by mapping performance and cost onto a **Cost-Quality plane**. The paper constructs a **Non-Decreasing Convex Hull (NDCH)** across Pareto-optimal parameter configurations. To summarize performance into a single metric, it computes the **Average Improvement in Quality (AIQ)**—the area under the convex hull normalized across cost bounds. Routers are benchmarked against individual models, a baseline "Zero router" (a probabilistic mix of models), and a hindsight **Oracle router**.

* **What can we use from the paper?**  
  * **Hindsight Oracle Formulation:** The formal definition of an Oracle that selects the cheapest model yielding a correct answer.
  * **NDCH & AIQ Metrics:** Mathematical tools to plot and measure cost-performance trade-off curves.
  * **Router Dynamics:** Predictive classifiers (KNN/MLP) often struggle to beat simple baseline ensembles on complex reasoning, while cascading routers require low evaluator error to reach Oracle efficiency.

---

### 2. RouterEval

* **What is done in the paper?**  
  The authors introduce **RouterEval**, an open-source benchmark constructed from over 200 million performance records across 12 popular LLM evaluation suites and 8,500+ LLMs. The study identifies a **"model-level scaling up" phenomenon**: as the candidate pool expands, a capable router significantly boosts total system accuracy, often outperforming the best single candidate model. They systematically evaluate classification routers (LinearR, MLPR, C-RoBERTa, MLC, PRknn) across different pool sizes and candidate pool compositions ("all-strong", "all-weak", "strong-to-weak").

* **How are the Routers evaluated in the paper?**  
  Routers are evaluated as classification models predicting optimal selection vectors using four metrics:
  1. **Original Accuracy:** Overall correctness of selected model outputs.
  2. **Reference Value:** Accuracy relative to a strong flagship reference model (e.g., GPT-4).
  3. **Best Single Model Value:** Accuracy relative to the top single candidate in the pool.
  4. **Classification Bias / Entropy:** Shannon entropy of prediction distributions to detect router bias toward specific models.

* **What can we use from the paper?**  
  * **Classification Entropy:** A metric to verify whether your router exhibits classification bias (e.g., over-routing to the large model).
  * **Relative Metric:** Measuring accuracy relative to the Best Single Model baseline.
  * **Pool Sizing Insight:** Empirical proof that performance gains grow fastest when pool size is between 2 and 10, justifying a 2-model (small vs. large) routing framework.

---

### 3. RouterArena

* **What is done in the paper?**  
  The authors introduce **RouterArena**, an open platform and live leaderboard for evaluating 12 commercial (Azure Model Router, NotDiamond, GPT-5) and open-source LLM routers (vLLM Semantic Router, GraphRouter, RouteLLM, CARROT, RouterDC) under a unified benchmark. They build an 8,400-query dataset structured around the **Dewey Decimal Classification (DDC)** system across 9 domains and 44 categories. Prompt difficulty is calibrated using Bloom's taxonomy and empirically categorized using 42 LLMs into Easy, Medium, and Hard bands.

* **How are the Routers evaluated in the paper?**  
  Routers are evaluated across **5 core dimensions** and ranked via an **Arena Score** (a weighted harmonic mean of accuracy and log-normalized cost):
  1. **Accuracy & Cost:** Average task correctness and dollar cost per 1k queries.
  2. **Routing Optimality:** Measured by **Optimal Selection Ratio** (% of queries where the router selects the cheapest correct model), Optimal Accuracy Ratio, and Optimal Cost Ratio.
  3. **Routing Robustness:** Consistency of routing choices under text noise (paraphrasing, typos).
  4. **Routing Latency:** Time overhead (ms) added by the router classifier.

* **What can we use from the paper?**  
  * **Optimal Selection Ratio:** The primary metric to evaluate how often your router selects the small model when it is sufficient.
  * **Difficulty Breakdown:** Adapting their Easy/Medium/Hard framework to a Low-Resource vs. High-Resource language split.
  * **Robustness & Overhead Protocols:** Testing router stability under perturbed inputs and verifying latency overhead (<100ms).

---

### 4. LLMRouterBench

* **What is done in the paper?**  
  The paper presents **LLMRouterBench**, a unified benchmark containing over 400k instances across 21 datasets and 33 models (20 lightweight ~7B models and 13 flagship models). It re-evaluates 10 representative routing baselines across Performance-Oriented and Performance-Cost Tradeoff paradigms. The study reveals that many complex routers yield similar accuracy, commercial routers often fail to beat the Best Single Model baseline, and a substantial gap to the Oracle remains due to **model-recall failures**.

* **How are the Routers evaluated in the paper?**  
  Evaluated using standardized metrics across both routing paradigms:
  * **Performance Metrics:** Average Accuracy AvgAcc, Gain@R (gain over random router), Gain@B (relative gain over Best Single Model), and Gap@O (relative distance to Oracle).
  * **Performance-Cost Metrics:** Peak accuracy gain PerfGain, maximum cost reduction without accuracy loss CostSave, and distance to the empirical Pareto frontier ParetoDist.
  * **Diagnostic Metrics:** Recall@k (whether any correct candidate model appears in the top-k ranked choices).

* **What can we use from the paper?**  
  * **Primary Project Metrics:** Adopt Gain@B and Gap@O as core evaluation targets.
  * **Model-Recall Failure Diagnosis:** Use Recall@k to test whether your language-blind baseline fails by ranking the small model below the threshold on low-resource queries.
  * **Practical Findings:** Backbone embedding models have minimal impact on routing quality, and small, well-curated model pools yield better outcomes than scaling ensemble size.