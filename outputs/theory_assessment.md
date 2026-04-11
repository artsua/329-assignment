# Theory Assessment: Unsupervised Learning, Clustering, K-means

Q1. [Remember | define]  
Define **unsupervised learning**.

Answer Key:  
Unsupervised learning is a type of learning in which the dataset contains features or inputs but no labels or target values. Its goal is to discover hidden structure, patterns, or groupings in the data.

Marking Scheme (Total: 4):  
- States that only features/inputs are available (2)  
- States that labels/targets are not available (1)  
- States that the goal is to discover structure/patterns/groupings (1)  
- Partial credit: award 1–3 marks for partially correct definitions missing one component.

Expected Response Depth:  
1–2 sentences.

---

Q2. [Remember | state]  
State the objective that K-means minimizes.

Answer Key:  
K-means minimizes the within-cluster sum of squared distances between data points and their assigned cluster centroids.

Marking Scheme (Total: 4):  
- Mentions minimization of within-cluster variation or compactness (1)  
- Mentions squared distances (1)  
- Mentions data points to centroids/means (2)  
- Partial credit: award 1–3 marks if the student identifies only a subset, such as “within-cluster distance.”

Expected Response Depth:  
1 sentence.

---

Q3. [Understand | explain]  
Explain why feature scaling is important before applying K-means.

Answer Key:  
K-means uses Euclidean distance to assign points to the nearest centroid. If one feature has a much larger numerical scale than the others, it dominates the distance calculation. As a result, clustering becomes driven mostly by that feature rather than by the overall structure of the data. Scaling helps each feature contribute more fairly to distance calculations.

Marking Scheme (Total: 6):  
- States that K-means relies on Euclidean distance (2)  
- Explains that larger-scale features dominate the distance (2)  
- Explains the consequence for clustering quality (1)  
- States that scaling balances feature influence (1)  
- Partial credit: award 1–5 marks depending on how many of the above ideas are clearly explained.

Expected Response Depth:  
3–4 sentences.

---

Q4. [Understand | compare]  
Compare **clustering** and **classification**.

Answer Key:  
Classification is a supervised learning task in which the model predicts known labels using labeled training data. Clustering is an unsupervised learning task in which the algorithm groups data points based on similarity without using labels. In classification, the categories are predefined. In clustering, the groups are discovered from the data itself.

Marking Scheme (Total: 6):  
- Identifies classification as supervised (2)  
- Identifies clustering as unsupervised (2)  
- Explains that classification uses known labels (1)  
- Explains that clustering discovers groups from the data itself (1)  
- Partial credit: award 1–5 marks for partially correct comparisons.

Expected Response Depth:  
3–5 sentences.

---

Q5. [Apply | calculate]  
A cluster contains the one-dimensional points 2, 4, and 8. Calculate the centroid of the cluster.

Answer Key:  
The centroid is the mean of the points.

\[
\mu = \frac{2 + 4 + 8}{3} = \frac{14}{3} \approx 4.67
\]

So, the centroid is \(14/3\) or approximately 4.67.

Marking Scheme (Total: 5):  
- States that centroid is the mean (1)  
- Correct substitution into the formula (2)  
- Correct final answer \(14/3\) or 4.67 (2)  
- Partial credit: if arithmetic is slightly wrong but method is correct, award up to 3 marks.

Expected Response Depth:  
2–3 lines with calculation.

---

Q6. [Apply | predict]  
A point \((4, 3)\) must be assigned to one of two centroids: \(C_1=(1,1)\) and \(C_2=(6,4)\). Using Euclidean distance, predict which cluster K-means will assign the point to.

Answer Key:  
Compute the distances.

Distance to \(C_1\):
\[
\sqrt{(4-1)^2 + (3-1)^2} = \sqrt{9 + 4} = \sqrt{13}
\]

Distance to \(C_2\):
\[
\sqrt{(4-6)^2 + (3-4)^2} = \sqrt{4 + 1} = \sqrt{5}
\]

Since \(\sqrt{5} < \sqrt{13}\), the point is assigned to cluster 2.

Marking Scheme (Total: 6):  
- Correct distance expression to \(C_1\) (2)  
- Correct distance expression to \(C_2\) (2)  
- Correct comparison of distances (1)  
- Correct final cluster assignment (1)  
- Partial credit: if the student uses squared distances correctly without square roots, full marks may still be awarded.

Expected Response Depth:  
4–6 lines with calculation.

---

Q7. [Analyze | diagnose]  
A team runs K-means with \(K=3\) on the same scaled tabular dataset five times. They obtain noticeably different cluster assignments across runs, even though the data and preprocessing did not change. Diagnose the most likely cause and explain the cause-effect relationship.

Answer Key:  
The most likely cause is **sensitivity to initialization**. K-means starts with initial centroid positions, and different random starting points can lead the algorithm toward different **local optima** even when the dataset and preprocessing remain unchanged. Because the data is already scaled, inconsistent scaling is unlikely to explain the variation, so the changing starting centroids are the stronger explanation. The cause-effect relationship is: **different initial centroids lead to different optimization paths, which can produce different final cluster boundaries and assignments**. This happens because K-means reduces the objective iteratively but does not guarantee the same global optimum on every run. A practical fix is to use **k-means++ initialization** and a larger **`n_init`** so that multiple runs are tried and the best solution is selected.

Marking Scheme (Total: 8):  
- Correctly diagnoses initialization sensitivity as the main cause (3)  
- Explains that different starting centroids can lead to different local optima (3)  
- Clearly links the cause to different final assignments across runs (1)  
- Suggests a valid mitigation such as k-means++ or multiple restarts (1)  
- Partial credit: award 1–7 marks depending on the clarity and completeness of the diagnosis.

Expected Response Depth:  
4–6 sentences with diagnosis and cause-effect reasoning.

---

Q8. [Analyze | differentiate]  
A dataset contains one large compact group, one small meaningful rare group, and several extreme outliers. After applying K-means, the rare group is merged into the large group, while one centroid moves toward the outliers. Differentiate the two technical reasons causing these two failures.

Answer Key:  
Two different issues are occurring. First, the rare group is being merged because K-means can struggle with cluster size imbalance; its centroid-based objective tends to favor larger dense groups and may fail to preserve a small but meaningful cluster. Second, one centroid moves toward the outliers because K-means is sensitive to extreme points; outliers can pull the mean away from the dense region of normal data. So the first failure is caused by imbalance in cluster size and representation, while the second is caused by the effect of outliers on the centroid.

Marking Scheme (Total: 8):  
- Correctly identifies the first issue as small-group/cluster-size imbalance (3)  
- Correctly identifies the second issue as outlier sensitivity (3)  
- Clearly differentiates the two causes (1)  
- Links each cause to the observed effect (1)  
- Partial credit: award 1–7 marks depending on whether both causes are separated clearly.

Expected Response Depth:  
5–7 sentences with explicit differentiation.

---

Q9. [Evaluate | justify]  
A practitioner must choose between K-means and DBSCAN for a numeric dataset whose clusters appear curved and irregular, and the dataset also contains noise points. Justify which method is more appropriate using criteria from the lecture.

Answer Key:  
DBSCAN is the more appropriate choice. K-means works best when clusters are compact, roughly spherical, and separable by distance to the mean. It also forces every point into a cluster and is sensitive to outliers. In contrast, DBSCAN is designed for density-based clustering, can detect irregularly shaped clusters, and can label noise points as outliers rather than forcing them into a group. Using the criteria of cluster shape and outlier handling, DBSCAN is better suited for this dataset.

Marking Scheme (Total: 8):  
- Chooses DBSCAN correctly (2)  
- Uses cluster-shape criterion correctly (2)  
- Uses outlier/noise-handling criterion correctly (2)  
- Justifies why K-means is less suitable (2)  
- Partial credit: award 1–7 marks if the recommendation is reasonable but incompletely justified.

Expected Response Depth:  
4–6 sentences with explicit criteria.

---

Q10. [Evaluate | assess]  
A bank clusters loan applicants using income patterns, transaction behavior, and repayment timing. The bank wants to automatically reject one cluster labeled “financially risky.” Assess this decision ethically and technically.

Answer Key:  
This decision is **not ethically or technically well justified**. Ethically, clustering can group applicants in ways that act as proxies for disadvantage, even if protected attributes are not explicitly included. For example, income patterns or transaction behavior may indirectly reflect socioeconomic background rather than actual creditworthiness in a fair and reliable way. Technically, K-means forces every applicant into a cluster, even when some applicants may not fit clearly into any meaningful group, and the result depends heavily on feature choice, scaling, initialization, and the chosen value of \(K\). That means the cluster label “financially risky” is not an objective truth, but a modeling outcome shaped by many design decisions. Since loan rejection is a high-stakes decision, the bank should not use cluster membership alone for automatic rejection. A more responsible approach would use clustering only for exploratory analysis, followed by stronger validation, fairness review, and human oversight before any lending decision.

Marking Scheme (Total: 10):  
- Identifies a specific ethical risk such as proxy bias or unfair disadvantage (3)  
- Identifies a technical limitation such as forced assignment, dependence on preprocessing, or instability (3)  
- Clearly assesses the proposed automatic rejection as unjustified or inappropriate (2)  
- Recommends responsible alternatives such as validation, fairness checks, or human review (2)  
- Partial credit: award 1–9 marks depending on the quality and balance of ethical and technical reasoning.

Expected Response Depth:  
5–8 sentences with judgment and justification.

---

Q11. [Create | design]  
Design a practical workflow for applying K-means to an online retail customer dataset that contains missing values, differently scaled numeric features, and possible outliers. Your workflow must include dataset preparation, preprocessing, model selection, evaluation, and interpretation.

Answer Key:  
A practical workflow would begin by loading the dataset, inspecting the columns, removing duplicates if needed, and selecting features that are meaningful for customer segmentation. Next, split the raw data into training, validation, and test sets before fitting any preprocessing steps. Then handle missing values using an appropriate imputation strategy, keep suitable numeric features, and scale them so that Euclidean distance is not dominated by larger-scale variables. Since outliers may distort centroids, inspect the data for extreme values and either treat them carefully or compare results before and after outlier handling. After preprocessing, train K-means on the training set using **k-means++** initialization and multiple restarts through a sufficiently large **`n_init`**. Try several values of \(K\), compare them using **inertia**, **silhouette score**, and **stability across runs**, and choose a clustering that is both reasonably stable and interpretable. Finally, interpret the clusters by examining centroid profiles, feature summaries, and cluster sizes, and check whether the discovered segments make business sense before using them in practice.

Marking Scheme (Total: 10):  
- Includes dataset preparation and relevant feature selection (2)  
- Includes handling of missing values and scaling (2)  
- Includes explicit treatment or checking of outliers (2)  
- Includes suitable K-means training and model selection strategy (2)  
- Includes interpretation and domain validation of results (2)  
- Partial credit: award marks for each valid workflow component even if the response is incomplete or not perfectly sequenced.

Expected Response Depth:  
6–8 sentences or a clearly structured stepwise plan.

---

Q12. [Create | propose]  
Propose a responsible deployment plan for using clustering in a hospital system to group patients for supportive care planning. Your plan must reduce the risks specific to clustering while still allowing useful analysis.

Answer Key:  
A responsible plan would treat clustering as an exploratory tool rather than an automatic decision system. The hospital should first choose features carefully and review whether any features may act as problematic proxies for sensitive or disadvantaged groups. The data should be preprocessed consistently, and multiple clustering runs or alternative methods should be compared to test stability. The hospital should validate whether the discovered groups are clinically meaningful rather than relying only on low inertia or a single metric. Any cluster labels should be non-stigmatizing, and no patient should be denied care based only on cluster membership. Final decisions should involve human oversight, documentation of limitations, and ongoing monitoring to detect drift or inequitable impact.

Marking Scheme (Total: 10):  
- Proposes clustering as exploratory rather than automatic for high-stakes use (2)  
- Includes bias mitigation or proxy-feature review (2)  
- Includes validation/stability checking (2)  
- Includes human oversight and non-exclusive decision use (2)  
- Includes documentation or ongoing monitoring (2)  
- Partial credit: award marks for each responsible safeguard proposed clearly.

Expected Response Depth:  
6–8 sentences with a coherent deployment plan.
