# Related work and how this study is positioned

Compiled on 2026-09-20 from verified sources; every link below was opened or returned by a search on that date. It answers one question in particular: **can a reviewer say that what we report is already known?** The short answer is that each ingredient exists in some literature, but not the combination, and not in this problem.

## 1. Imbalanced regression: the same intuition, different lever

The intuition that a target covered unevenly gives poor models in the rare regions is an established field.

- Branco, Torgo & Ribeiro, *A Survey of Predictive Modeling on Imbalanced Domains*, ACM Computing Surveys, 2016. [dl.acm.org](https://dl.acm.org/doi/10.1145/2907070)
- Branco, Torgo & Ribeiro, *SMOGN: a Pre-processing Approach for Imbalanced Regression*, PMLR 74, 2017. [proceedings.mlr.press](https://proceedings.mlr.press/v74/branco17a.html)
- Yang, Zha, Chen, Wang & Katabi, *Delving into Deep Imbalanced Regression*, ICML 2021. [proceedings.mlr.press](https://proceedings.mlr.press/v139/yang21m.html) · [arXiv:2102.09554](https://arxiv.org/abs/2102.09554)
- Steininger, Kobs et al., *Density-based weighting for imbalanced regression* (DenseWeight), Machine Learning 110, 2021. [semanticscholar.org](https://www.semanticscholar.org/paper/Density-based-weighting-for-imbalanced-regression-Steininger-Kobs/5e35f17cc6e37ab83f3019d7e86b7dd13a892c55)

**Difference.** All of these act on a dataset that already exists: they re-sample or re-weight it. Here the data do not exist yet, and each point costs a solver run. The lever is *which simulations to run*, at a fixed budget. Re-weighting cannot create a point that was never computed.

## 2. Output space-filling designs: the closest prior art

- Wang, Generale, Kalidindi & Joseph, *Sequential Designs for Filling Output Spaces*, **Technometrics 66(1), 2024** (online 2023). [doi:10.1080/00401706.2023.2231042](https://doi.org/10.1080/00401706.2023.2231042) · [arXiv:2305.07202](https://arxiv.org/abs/2305.07202). Cite the journal version.
  From the abstract: space-filling designs fill the space of inputs, "however, in certain applications such as inverse design or feature-based modeling, the aim is to fill the response or feature space". The design is adaptive and model-free, and the examples show advantages "over the traditional input space-filling designs".
- Lu & Anderson-Cook, *Input-response space-filling designs*, Quality and Reliability Engineering International (accepted manuscript LA-UR-20-25681). [osti.gov](https://www.osti.gov/pages/biblio/1853923-input-response-space-filling-designs)
  These designs balance input space-filling against response space-filling, measured by minimum pairwise distances in each space.

**Difference.** The motivation there is inverse design, feature-based modeling and coverage as such. Our question is different, and narrower: at a fixed budget of expensive analyses, what does response-space uniformity do to **forward predictive accuracy in a region of engineering interest**, and why. We also decompose the effect (principle, screener, within-bin rule, learner), which a design-oriented paper has no reason to do.

**For the paper.** These two are the works to cite when stating that output space-filling is recommended where the aim is to represent rare behaviour. Our result qualifies that recommendation: the rare region and the region of interest need not be the same, and in this problem they are not.

## 3. Active learning for regression: output diversity alone is known to be weak

- Wu, Lin & Huang, *Active Learning for Regression Using Greedy Sampling*, Information Sciences 474, 2019, 90–105. [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680) · [arXiv:1808.04245](https://arxiv.org/abs/1808.04245) · [code](https://github.com/drwuHUST/iGS)
  GSy increases diversity in the output space, GSx in the input space, and iGS in both. The reported weakness of GSy is that it "can be unreliable if the initial model is poor".

**Relation.** This is the literature closest to our screener result: a design driven by predicted output inherits the screener's errors. Our C-log to C-seq comparison measures exactly how much retraining the screener recovers, in a simulation-budget setting rather than on UCI data.

## 4. Adaptive sampling for surrogates: no systematic win over space-filling

- Liu, Ong & Cai, *A survey of adaptive sampling for global metamodeling in support of simulation-based complex engineering design*, Structural and Multidisciplinary Optimization, 2018. [link.springer.com](https://link.springer.com/article/10.1007/s00158-017-1739-8)
- Fuhg, Fau & Nackenhorst, *State-of-the-Art and Comparative Review of Adaptive Sampling Methods for Kriging*, Archives of Computational Methods in Engineering, 2021. [link.springer.com](https://link.springer.com/article/10.1007/s11831-020-09474-6) · [code](https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling)
- Su, Wang, Bursi & Broccardo, *Surrogate modeling for probability distribution estimation: uniform or adaptive design?*, **Reliability Engineering & System Safety 261, 2025**, 111059. [doi:10.1016/j.ress.2025.111059](https://doi.org/10.1016/j.ress.2025.111059) · [arXiv:2404.07323](https://arxiv.org/abs/2404.07323). Cite the journal version.
  Across 16 benchmarks and 1 920 analyses they conclude that active-learning techniques "cannot provide a systematic improvement compared with uniform designs", and that the right choice depends on the problem's local nonlinearity, the target accuracy and the budget.

**Relation.** This is the strongest precedent for a result of our shape, and it is an ally rather than a threat: it shows that a rigorous negative comparison against space-filling designs is publishable and useful. Their target is the full distribution function of the response; ours is accuracy in a region defined by engineering relevance. Both lines of evidence point the same way: space-filling in the inputs is hard to beat, and the exceptions are local.

## 5. Geotechnical surrogates: two close works, neither the same question

Two papers are close enough that the claim "sampling has not been studied for slope-stability surrogates" would be wrong. Both must be cited and discussed head-on.

- **Shakya, Schmüdderich, Machacek, Prada-Sarmiento & Wichtmann**, *Influence of Sampling Methods on the Accuracy of Machine Learning Predictions Used for Strain-Dependent Slope Stability*, Geosciences 14(2), 44, 2024. [doi:10.3390/geosciences14020044](https://doi.org/10.3390/geosciences14020044)
  From the abstract: the influence of the sampling strategy on model performance "has received comparatively limited attention"; the authors train ML models for strain-dependent slope stability (hypoplasticity) under **six sampling methods**, and introduce a new one to represent the **low stress range** better. They report that the sampling method markedly affects the shear-stress predictions at low stress and the resulting factor of safety.
  **Difference.** Their surrogate stands in for the constitutive response, and their "hard region" is the low stress range of that response, sampled in the input space. They do not balance the distribution of the FOS itself, do not work at a fixed budget of analyses against an LHS baseline, and do not report where the error moves along the FOS axis. But they do establish, in our field, that the sampling strategy matters — so our contribution must be the *response-space allocation and its trade-off*, not "sampling matters".

- **Mazloom, Shafieezadeh, Hur, Jung, Ha & Hahm**, *BARL: Batch active regression learning via query-by-committee with application to slope factor of safety prediction*, Applied Soft Computing **188 (2026) 114412** (accepted 1 Dec 2025). [doi:10.1016/j.asoc.2025.114412](https://doi.org/10.1016/j.asoc.2025.114412)
  **The closest work found. Read in full on 2026-09-20; everything below is from the paper itself.**

  Batch active learning for a continuous FOS target. A committee of five LightGBM models, each trained on a different 90–95% subset, predicts the FOS of an LHS candidate pool; a batch of the most informative candidates is labelled by FLAC and added, and the committee is retrained. The generator is FLAC8 with gwflow, Mohr–Coulomb, eight inputs including porosity, initial saturation and initial pore pressure.

  **What it shares with us.** More than the title suggests, and all of it must be acknowledged:
  - *A screener that predicts the response in order to steer selection.* Their disagreement score (eq. 2) is the committee variance σᵢ **gated by a range term**: weight 1 when the committee's mean prediction μᵢ falls inside the acceptable FOS range, and an exponential penalty outside it. So candidates are filtered by a *predicted* FOS, exactly the idea behind our screener — prior art for the mechanism, not something we invented.
  - *A diversity rule inside the batch.* A cosine-similarity threshold β rejects a candidate too close to one already chosen — the same role our max-min within-bin rule plays.
  - *Balance of the FOS distribution presented as the gain.* §5.4: "Initially, the dataset was skewed toward lower FoS values, with higher values (4–6) under-represented. After BARL, the FoS distribution became more balanced." They call this "the greatest improvement".
  - *The same domain limits.* §3.3: the training set is LHS, then **filtered to FOS ∈ [0.3, 6]**, samples outside removed, final N = 300. And §2.2: "while most practical scenarios involve FoS values between 0.9 and 2, our goal is to construct the model to predict a broader range, extending from 0.3 to 6."

  **Differences, in order of how much they matter to us:**
  1. **The comparison is not budget-fair, and this is the decisive gap.** §5.5 trains two committees: G1 on the *final* dataset and G2 on the *initial* 300 samples. At iteration 512 the final dataset holds **5 410 samples** (Fig. 4: sample count = iteration × 10 + 300). The headline "75% reduction in test MSE, 60% in MAE" is therefore **5 410 labelled samples against 300** — roughly an 18-fold budget difference — not BARL against random selection at the same budget. §5.1 claims the design "isolates the contribution of the BARL framework"; with the sample count varying by 18×, it does not. **No arm anywhere in the paper spends the same number of solver runs by a non-adaptive rule.** That comparison is the one our study runs.
  2. **Balance is a by-product, not the objective.** The criterion is variance × range-gate. Nothing fills a deficit or targets a density; the flatter FOS histogram is observed after the fact and never tested as a cause.
  3. **No decomposition along the FOS axis.** Everything is aggregate R², MSE and MAE on the FOS scale, over the whole range. Fig. 6 shows the error growing with FOS, and §5.4 notes that the initial model is weak in 4–6, but no zone metric is computed and no low-FOS region is evaluated separately.
  4. **The direction of rarity is the opposite of ours.** Their filtered LHS set is skewed *low* (Table 1: 25% at 1.29, median 2.09, mean 2.40), so the under-represented region is 4–6. Ours is skewed *high* (46–48% in 2–6, 10–11% in 0.5–1). Same family of designs, mirror-image imbalance — which is evidence for our §4.2 point that *which* region is rare is a property of the parameter domain, not a general law, and a caution against generalising either result.
  5. Stopping is by convergence over 512 iterations, not by a fixed budget; each iteration re-tunes every committee member with `BayesSearchCV`.

  **For the paper.** Cite BARL as the closest geotechnical work, credit it with the predicted-FOS gate and the batch-diversity rule, and state the gap plainly: it demonstrates that adaptive sampling with 18× more data beats a small fixed dataset, which is not in question; what is in question is the allocation of a *fixed* budget, and that is what we measure.

Other geotechnical ML surrogates follow the usual pattern — a parametric study or a random-field generator, a 60/20/20 split, and a comparison of learners — without treating the allocation of the analysis budget across the FOS range as the object of study:

- *A Surrogate Model for the Rapid Prediction of Factor of Safety in Slopes with Spatial Variability*, Mathematics 13, 2025. [doi.org/10.3390/math13101604](https://doi.org/10.3390/math13101604)
- *A pre-trained deep-learning surrogate model for slope stability analysis with spatial variability*, Soils and Foundations, 2023. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0038080623000501)
- *Development of machine learning surrogate models for slope stability prediction using AI techniques*, Scientific Reports, 2025. [nature.com](https://www.nature.com/articles/s41598-025-33324-9)

Reliability-oriented work (AK-MCS and its descendants) targets the limit state FOS = 1 in order to estimate a probability of failure: a different objective and a different metric, and the reason the Georisk venue was set aside.

**Conclusion of the search.** The ingredients are all in print — that sampling matters here (Shakya), that a predicted FOS can gate selection and that the resulting FOS histogram flattens (BARL), that response-space filling is recommended (Wang, Lu), and that adaptive designs do not systematically beat uniform ones (Su). What no study does is hold the **budget** fixed and ask where the accuracy goes. BARL, the closest, compares 5 410 samples against 300. That is the opening, and it is a narrow one: the contribution is the budget-fair design and the decomposition along the FOS axis, not the idea of shaping the response distribution.

Repeat the search with the final wording before submission, and check the citation lists of Wang et al., Su et al. and BARL — in particular Mazloom et al., *Eng. Appl. Artif. Intell.* 165 (2026) 113998, by the same group, which we have not read.

## 6. How to phrase the contribution so the "already known" objection is answered

1. **Do not claim that sampling is unstudied here.** Shakya et al. (2024) already show that the sampling strategy matters for slope-stability surrogates, and BARL (2026) already applies active learning to a continuous FOS target, gates it on a predicted FOS, and reports a flatter FOS histogram as its main gain. The claim is narrower and defensible: *at equal simulation budget, does balancing the distribution of the continuous FOS improve regional accuracy against a parameter-space space-filling design?* The answer we report is that the main effect is not a global gain but a redistribution of accuracy between stability regimes, which depends on the density actually achieved, on the learner, and on sequential adaptation.
2. **Make budget-fairness the load-bearing word.** It is what separates us from BARL, whose reported gain compares 5 410 labelled samples against 300. Every arm here spends the same number of solver runs; the ablations (C₀, C₀-log, C₀-seq) exist so that the budget cannot leak. Say it in the Introduction, not only in Methods.
3. **State the recommendation being tested,** with its citations (§2): output space-filling is advised where rare behaviour must be represented.
4. **Show that the premise fails here,** with numbers: under LHS the decision zone 1–2 already holds 30–34% of the points, more than a log-uniform target would give it; only 0.5–1 is rare.
5. **Report the effect as relocation,** by zone, at a fixed budget, with the ablations.
6. **Decompose it:** the principle, the screener (C-log against C-seq), the within-bin rule (max-min against random) and the learner.
7. **Keep the negative part explicit,** and cite Su et al. as the precedent that such a result is informative rather than empty.
8. **State the limits:** one generator (xslope, Bishop simplified), homogeneous drained slopes, two tuned tree-ensemble families, and conclusions conditional on the seed and the test sets.
