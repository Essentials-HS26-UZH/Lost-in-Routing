# Language-Aware LLM Routing: Project Decisions

> In this file, we can document our key decisions for the project. The current recommendations should be discussed and may be changed.

## 1. Datasets (Multilingual & Deterministic)

* **Global-MMLU:** General knowledge (57 subjects) across 42 languages. Format: 4-option MCQs. Highly reliable for cross-lingual knowledge.

* **Belebele:** Reading comprehension across 122 languages. Format: 4-option MCQs. Features parallel text, acting as an excellent control variable.

* **MGSM:** Multi-step math reasoning across 10 diverse languages. Format: Exact-match numerical.

## 2. Language Selection (Ensuring dataset overlap)

* **High-Resource:** **Spanish (es)**. Massive pre-training representation; provides a strong baseline where the small model is frequently sufficient.

* **Low-Resource:** **Swahili (sw)**. Underrepresented in pre-training corpora; provides a rigorous test for the router to default to the larger model.

## 3. Model Selection (Same-family to isolate scale)

* **Option A: Qwen2.5 Family** (`Qwen2.5-7B-Instruct` & `Qwen2.5-72B-Instruct`). Currently holds state-of-the-art open-weight performance for multilingual tasks.

* **Option B: Llama-3.1 Family** (`Llama-3.1-8B-Instruct` & `Llama-3.1-70B-Instruct`). The most widely adopted open-weight baseline in academic literature.

## 4. Evaluation Metrics

*Reference: Metrics are formalized based on "LLMRouterBench-metrics-suite.md".*

To evaluate the language-aware router against baselines like the Best Single Model ($B$), Random Router ($R$), and the hindsight Oracle ($O$), we will track the following core metrics:

* **Performance Metrics:**
  * **Average Accuracy ($\text{AvgAcc}$):** The overall mean correctness across all evaluated language subsets.
  * **Gain over Best Single Model ($\text{Gain@B}$):** The relative percentage accuracy improvement achieved by the router over always using the single best candidate model.
  * **Gain over Random ($\text{Gain@R}$):** The relative accuracy improvement compared to a baseline that selects models uniformly at random.
  * **Gap to Oracle ($\text{Gap@O}$):** The distance between the router's accuracy and the theoretical maximum of the ideal hindsight Oracle (ideal value is $0\%$).

* **Cost-Tradeoff Metrics:**
  * **Peak Performance Gain ($\text{PerfGain}$):** The maximum relative accuracy improvement achievable over $B$ across all tunable threshold configurations $\theta$.
  * **Maximum Cost Savings ($\text{CostSave}$):** The maximum percentage reduction in total inference cost achievable without dropping average accuracy below $B$.
  * **Distance to Pareto Frontier ($\text{ParetoDist}$):** Measures how close the router's (cost, accuracy) operating points are to the optimal empirical Pareto frontier.

* **Diagnostic Metrics:**
  * **$\text{Recall@k}$:** Evaluates whether a correct candidate model is successfully ranked within the top-$k$ predictions by the router.