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

- Wang, Generale, Kalidindi & Joseph, *Sequential Designs for Filling Output Spaces*, 2023. [arXiv:2305.07202](https://arxiv.org/abs/2305.07202)
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
- Su, Wang, Bursi & Broccardo, *Surrogate modeling for probability distribution estimation: uniform or adaptive design?*, 2024. [arXiv:2404.07323](https://arxiv.org/abs/2404.07323)
  Across 16 benchmarks and 1 920 analyses they conclude that active-learning techniques "cannot provide a systematic improvement compared with uniform designs", and that the right choice depends on the problem's local nonlinearity, the target accuracy and the budget.

**Relation.** This is the strongest precedent for a result of our shape, and it is an ally rather than a threat: it shows that a rigorous negative comparison against space-filling designs is publishable and useful. Their target is the full distribution function of the response; ours is accuracy in a region defined by engineering relevance. Both lines of evidence point the same way: space-filling in the inputs is hard to beat, and the exceptions are local.

## 5. Geotechnical surrogates: no equivalent study found

Searches for slope-stability surrogates combined with a designed training-set distribution over FOS returned the usual pattern: a parametric study or a random-field generator, a dataset split 60/20/20, and a comparison of learners. Examples:

- *A Surrogate Model for the Rapid Prediction of Factor of Safety in Slopes with Spatial Variability*, Mathematics 13, 2025. [doi.org/10.3390/math13101604](https://doi.org/10.3390/math13101604)
- *A pre-trained deep-learning surrogate model for slope stability analysis with spatial variability*, Soils and Foundations, 2023. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0038080623000501)
- *Development of machine learning surrogate models for slope stability prediction using AI techniques*, Scientific Reports, 2025. [nature.com](https://www.nature.com/articles/s41598-025-33324-9)

None of them treats the allocation of the analysis budget across the FOS range as the object of study, and none reports accuracy by FOS zone under a fixed budget. Reliability-oriented work (AK-MCS and its descendants) targets the limit state FOS = 1 for the purpose of estimating a probability of failure, which is a different objective and a different metric; it is also why the Georisk venue was set aside.

**Conclusion of the search.** No 1:1 overlap was found. Before submission, repeat the search with the final wording of the thesis, and check the citation lists of the Wang et al. and Su et al. papers.

## 6. How to phrase the contribution so the "already known" objection is answered

1. **State the recommendation being tested,** with its citations (§2): output space-filling is advised where rare behaviour must be represented.
2. **Show that the premise fails here,** with numbers: under LHS the decision zone 1–2 already holds 30–34% of the points, more than a log-uniform target would give it; only 0.5–1 is rare.
3. **Report the effect as relocation,** by zone, at a fixed budget, with the ablations.
4. **Decompose it:** the principle, the screener (C-log against C-seq), the within-bin rule (max-min against random) and the learner.
5. **Keep the negative part explicit,** and cite Su et al. as the precedent that such a result is informative rather than empty.
6. **State the limits:** one generator (xslope, Bishop simplified), homogeneous drained slopes, two tuned tree-ensemble families, and conclusions conditional on the seed and the test sets.
