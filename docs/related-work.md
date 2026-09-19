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

- **Mazloom, Shafieezadeh, Hur, Jung, Ha & Hahm**, *BARL: Batch active regression learning via query-by-committee with application to slope factor of safety prediction*, Applied Soft Computing 188, 114412, 2025. [doi:10.1016/j.asoc.2025.114412](https://doi.org/10.1016/j.asoc.2025.114412)
  Batch active learning for a continuous FOS target, selecting by committee disagreement.
  **Difference.** The criterion is model uncertainty, not the distribution of the target; the paper reports an overall gain over non-adaptive sampling, not a decomposition by FOS region.
  **Verification note (2026-09-20):** the abstract could not be retrieved from OpenAlex or Semantic Scholar, and the publisher page is not accessible without a subscription. Title, authors, journal, volume and year are confirmed; everything else about this paper must be read in full before it is described in the manuscript. It is the closest work found, so it deserves a paragraph of its own in Related Work.

Other geotechnical ML surrogates follow the usual pattern — a parametric study or a random-field generator, a 60/20/20 split, and a comparison of learners — without treating the allocation of the analysis budget across the FOS range as the object of study:

- *A Surrogate Model for the Rapid Prediction of Factor of Safety in Slopes with Spatial Variability*, Mathematics 13, 2025. [doi.org/10.3390/math13101604](https://doi.org/10.3390/math13101604)
- *A pre-trained deep-learning surrogate model for slope stability analysis with spatial variability*, Soils and Foundations, 2023. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0038080623000501)
- *Development of machine learning surrogate models for slope stability prediction using AI techniques*, Scientific Reports, 2025. [nature.com](https://www.nature.com/articles/s41598-025-33324-9)

Reliability-oriented work (AK-MCS and its descendants) targets the limit state FOS = 1 in order to estimate a probability of failure: a different objective and a different metric, and the reason the Georisk venue was set aside.

**Conclusion of the search.** No study asks our question — response-space uniformity against parameter-space space-filling, at equal budget, with the error decomposed by FOS region. Repeat the search with the final wording before submission, and check the citation lists of Wang et al., Su et al. and BARL.

## 6. How to phrase the contribution so the "already known" objection is answered

1. **Do not claim that sampling is unstudied here.** Shakya et al. (2024) already show that the sampling strategy matters for slope-stability surrogates, and BARL (2025) already applies active learning to a continuous FOS target. The claim is narrower and defensible: *at equal simulation budget, does balancing the distribution of the continuous FOS improve regional accuracy against a parameter-space space-filling design?* The answer we report is that the main effect is not a global gain but a redistribution of accuracy between stability regimes, which depends on the density actually achieved, on the learner, and on sequential adaptation.
2. **State the recommendation being tested,** with its citations (§2): output space-filling is advised where rare behaviour must be represented.
3. **Show that the premise fails here,** with numbers: under LHS the decision zone 1–2 already holds 30–34% of the points, more than a log-uniform target would give it; only 0.5–1 is rare.
4. **Report the effect as relocation,** by zone, at a fixed budget, with the ablations.
5. **Decompose it:** the principle, the screener (C-log against C-seq), the within-bin rule (max-min against random) and the learner.
6. **Keep the negative part explicit,** and cite Su et al. as the precedent that such a result is informative rather than empty.
7. **State the limits:** one generator (xslope, Bishop simplified), homogeneous drained slopes, two tuned tree-ensemble families, and conclusions conditional on the seed and the test sets.
