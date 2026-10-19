# Sokolov et al. (2025): prompts transcribed from Appendices B and C

Source: Sokolov, Sabelli, Azadie Faraz, Li & Seco, "Toward Automating Causal Discovery in Financial
Markets and Beyond," Transactions of ADIA Lab, ch. 14 (2025), pp. 385–390, Figs. B14.1–B14.7 and C14.1–C14.3.
Transcribed verbatim from the figure images, typos included. `[...]` = text the authors omitted in the paper.
Model in the paper: GPT-4, with outputs parsed into JSON via OpenAI Function Calling.

---

## Part 1 — Clustering (Appendix B)

Purpose: split a large feature set (100+) into small groups, so each group's DAG fits in one prompt.

### B1. Prompt 1a — initial clustering into groups G1 (Fig. B14.1)

**system:**
You are an expert analyst in finance, accounting and economics, specializing in clustering features related to finance, economics, investing, banking, etc.

**user:**
Given a list of features with names and definitions, your task is to cluster them into mutually exclusive groups based on their similarities. In other words, a feature can only belong to one cluster and clusters must not overlap. The clustering should be exhaustive (i.e. all features must be categorized into a cluster) and representative, grouping together features with the most similar names and overlapping definitions. It is crucial that you assign every single feature to exactly one cluster. Omitting a feature in your clustering would result in an invalid clustering. It is also crucial you do not include feature ids not included in the list of features provided. Provide arrays with the feature ids for each cluster. These arrays represent the features assigned to each cluster. Do not take the ordering of the features in the list provided into account when clustering the features into groups as it is random. You must solely focus your clustering based on the definitions of each features and their similarities. Each cluster must be contain at most 25 features. You must also return a number of distinct clusters between 6 and 15. It is crucial you remain within these bounds. Each feature is formatted as follows:

{Feature id: id, Feature Name: name, Feature definition: definition}

**Here is the list of features:**

{Feature id:1, Feature Name: Firm age, Feature Definition: Age of the company (i.e. how long since the company has been created)}

{Feature id:2, Feature Name: Liquidity of book assets, Feature Definition: Ortiz-Molina and Phillips Liquidity (i.e. Cash and Short-term Investments + Current Operating Assets plus total assets minus current assets minus intangible assets) scaled by total assets lagged by a year}

[...]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains an id for the cluster, and the ids of the features mapped to the cluster. Here is an example of such an output:

**Example output:**
{cluster id: 0, feature_ids: [1, 3, 7]}
{cluster id: 1, feature_ids [2, 19, 22, 23, 27, 56, 81]}
[...]

### B2. Prompt 1b — name and describe each group in G1 (Fig. B14.2)

**system:** (same as B1)

**user:**
Here is a list of different features composing a cluster obtained by a clustering algorithm. Each feature will be accompanied by its definition using the following format:

{Feature Name: name, Feature Definition: definition}

Here is the list of features composing the cluster and their definitions:

{Feature Name: Amihud Measure, Feature Definition: current Amihud Measure is a measure of illiquidity, which is the daily ratio of absolute stock return to its dollar volume. (i.e. Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price)}

{Feature Name: The high-low bid-ask spread, Feature Definition: High-low bid ask estimator from Daily High and Low Prices of the previous 21 trading days (i.e. the previous month)}

{Feature Name: Dollar trading volume, Feature Definition: Dollar Volume in other words Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price over the previous 6 months}

{Feature Name: Coefficient of variation for dollar trading volume, Feature Definition: Coefficient of variation for dollar trading volume (i.e. Dollar Volume in other words Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price) over previous 6 months.}

Generate a name and description for this cluster of data science features from finance, accounting and economics, emphasizing the commonalities and overlaps between feature names and definitions. Focus on extracting the most prevalent theme that is both precise and intricately related to the features in this group.

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains a name and description for the cluster. Here is the example name and description obtained for this example cluster:

**Example output:**
Cluster Name: Liquidity Metrics,
Cluster Description: This cluster encompasses features that provide insights into the liquidity of a stock. It includes measures such as the Amihud Measure, which gauges illiquidity by comparing absolute stock return to its dollar volume. The high-low bid-ask spread, derived from daily high and low prices over the previous month, also contributes to understanding liquidity conditions. The cluster further includes the dollar trading volume, calculated as the product of the total number of shares sold and the stock share price over the previous six months. The coefficient of variation for this dollar trading volume, which provides a measure of volatility, is also part of this cluster. Overall, these features collectively provide a comprehensive view of a stock's liquidity status and its fluctuations over time.

### B3. Prompt 2a — split each group into sub-groups Sg (Fig. B14.3)

**system:** (same as B1)

**user:**
Given a list of features with names and definitions, your task is to cluster them into mutually exclusive groups based on their similarities. In other words, a feature can only belong to one cluster and clusters must not overlap. The clustering should be exhaustive (i.e. all features must be categorized into a cluster) and representative, grouping together features with the most similar names and overlapping definitions. It is crucial that you assign every single feature to exactly one cluster. Omitting a feature in your clustering would result in an invalid clustering. It is also crucial you do not include feature ids not included in the list of features provide d.Provide arrays with the feature ids for each cluster. These arrays represent the features assigned to each cluster. Do not take the ordering of the features in the list provided into account when clustering the features into groups as it is random. You must solely focus your clustering based on the definitions of each features and their similarities.You must produce granular clusters with distinct themes. Nevertheless, this does not mean you must return clusters containing a single feature. Each feature is formatted as follows:

{Feature id: id, Feature Name: name, Feature definition: definition}

**Here is the list of features:**

{Feature id:4, Feature Name: Amihud Measure,
Feature Definition: current Amihud Measure is a measure of illiquidity, which is the daily ratio of absolute stock return to its dollar volume. (i.e. Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price)}

{Feature id:16, Feature Name: The high-low bid-ask spread,
Feature Definition: High-low bid ask estimator from Daily High and Low Prices of the previous 21 trading days (i.e. the previous month)}

{Feature id:35, Feature Name: Dollar trading volume,
Feature Definition: Dollar Volume in other words Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price over the previous 6 months}
[....]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains a name and description for the cluster. Here is the example name and description obtained for this example cluster:
*(sic — the paper repeats B2's caption here, but the example output is a list of sub-clusters)*

**Example output:**
Features in subcluster 1:
["Amihud Measure", "Dollar trading volume", "Coefficient of variation for dollar trading volume", [....]]

Features in subcluster 2:
["The high-low bid-ask spread", [....]]

### B4. Prompt 2b — name and describe each sub-group (Fig. B14.4)

**system:** (same as B1)

**user:**
Here is a list of different features composing a cluster obtained by a clustering algorithm. Each feature will be accompanied by its definition using the following format:

{Feature Name: name, Feature Definition: definition}

**Here is the list of features composing the cluster and their definitions:**

{Feature Name: Amihud Measure, Feature Definition: current Amihud Measure is a measure of illiquidity, which is the daily ratio of absolute stock return to its dollar volume. (i.e. Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price)}

{Feature Name: Dollar trading volume, Feature Definition: Dollar Volume in other words Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price over the previous 6 months}

{Feature Name: Coefficient of variation for dollar trading volume, Feature Definition: Coefficient of variation for dollar trading volume (i.e. Dollar Volume in other words Trading volume (total number of shares of a stock sold during a period) mutiplied by stock share price) over previous 6 months.}

Generate a name and description of at most 50 words for this cluster of data science features from finance, accounting and economics, emphasizing the commonalities and overlaps between feature names and definitions. Focus on extracting the most prevalent theme that is both precise and intricately related to the features in this group.

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains a name and description for the subcluster. Here is the example name and description obtained for this example subcluster:

**Example output:**
SubCluster Name: Liquidity and Trading Volume Analysis
SubCluster Description: This cluster focuses on the analysis of stock liquidity and trading volume, encompassing measures of illiquidity, dollar trading volume, and its variation over a six-month period.

### B5. Prompt 3a — re-cluster all sub-groups into final groups Gf (Fig. B14.5)

**system:** (same as B1)

**user:**
Given a list of groups of features with names, definitions and sizes, your task is to cluster them into mutually exclusive clusters based on their similarities. In other words, a group can only belong to one cluster and clusters must not overlap. The size of each group corresponds to the number of features included in each respective group. The clustering should be exhaustive (i.e. all groups of features must be clustered) and representative, grouping together groups of features with the most similar names and overlapping definitions. You must solely focus your clustering based on the definitions of each group of features and their similarities. It is crucial you do not forget to cluster some of the features. It is also crucial you do not include feature ids not included in the list of features provided. Provide arrays with the group ids for each cluster. These arrays represent the groups assigned to each cluster. Do not take the ordering of the groups in the list provided into account when clustering the features into groups as it is random. Each cluster must be contain at most 25 features and must contain at least 5 features. To calculate the size of each cluster, you must sum the sizes of each group of features included in each respective cluster. Each group is formatted as follows:

{SubCluster id: id, Group Name: Name, Group Description: description,, Group Size: size}

**Here is the list of features:**

{SubCluster id:1, Group Name: Company Longevity, Group Description: This cluster focuses on the temporal aspect of companies, specifically their age. It encapsulates the duration since the establishment of a firm, providing insights into its longevity and experience in the market., Group Size: 1}

{SubCluster id:2, Group Name: Asset Liquidity Metrics, Group Description: This cluster focuses on the liquidity of both book and market assets, including cash and short-term investments. It emphasizes the scaling of these assets by total assets or market assets, providing a comprehensive view of a company's liquidity status., Group Size: 3}

{SubCluster id:3, Group Name: Operating Working Capital Dynamics, Group Description: This cluster focuses on the yearly changes in operating assets, liabilities, and working capital, all scaled by total assets. It captures the dynamics of a company's short-term financial health and operational efficiency., Group Size: 3}

[...]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON output consists of an array of arrays each containing the subcluster ids (i.e. Group id) belonging to each respective cluster.

**Example output:**
[Cluster id: 1, SubCluster ids: [1, 7, 12]]
[Cluster id: 2, SubCluster ids: [2, 16, 19, 20, 24]]
[...]

### B6. Prompt 3b — name and describe each final group (Fig. B14.6)

**system:** (same as B1)

**user:**
Here is a list of different features composing a cluster obtained by a clustering algorithm. Each feature will be accompanied by its definition using the following format:

{Feature Name: name, Feature Definition: definition}

**Here is the list of features composing the cluster and their definitions:**

{Feature Name: Change in current operating assets, Feature Definition: change in current operating assets (i.e. Current Assets minus Cash and Short-Term Investments) over 1 year scaled by current total assets}

{Feature Name: Change in current operating liabilities, Feature Definition: change in current operating liabilities (i.e. Current liabilities minus Debt in current liabilities) over 1 year scaled by current total assets}

{Feature Name: Change in current operating working capital, Feature Definition: Operating Working Capital (i.e. current operating assets minus current operating liabilities) Change over the previous year scaled by current total assets.}

[...]

{Feature Name: Change PPE and Inventory, Feature Definition: difference in Total (Gross) Property, Plant and Equipment (i.e. the cost and/or valuation of tangible fixed assets used in the production of revenue) Less Inventories over a year scaled by total assets lagged 1 year}

Generate a name and description for this cluster of data science features from finance, accounting and economics, emphasizing the commonalities and overlaps between feature names and definitions. Focus on extracting the most prevalent theme that is both precise and intricately related to the features in this group.

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains a name and description for the final cluster generated. Here is the example name and description obtained for this example final cluster:

**Example output:**
Cluster Name: Operating Assets and Liabilities Dynamics,
Cluster Description: This cluster focuses on the changes and growth in operating assets and liabilities, both current and non-current, over a period of one year, with all changes scaled by total assets. It includes the dynamics of inventory, which is a key component of current operating assets. The cluster also captures the changes in net operating assets, which is the difference between total operating assets and liabilities. Additionally, it includes the change in Property, Plant, and Equipment (PPE) less inventories, providing a comprehensive view of the company's operating capital and its changes over time.

### B7. Prompt 4 — final merge check (Fig. B14.7)

**system:** (same as B1)

**user:**
Given a list of mutually exclusive clusters of features with names, descriptions and sizes, your task is to determine if some clusters should be combined together based on the descriptions and similarities. The size of each cluster corresponds to the number of features included in each respective cluster. You goal in other words is to determine if the clustering is too granular. You must combine clusters who have extremely similar definitions until all clusters and definition are completely distinct from one another. If there are some clusters that need to be combined, return True and provide arrays with the cluster ids of clusters which need to be combined. These arrays represent the groups assigned to each cluster. It is crucial the clustering remains mutually exclusive after applying your combinations. It is possible for the clusters to already be sufficiently distinct. In this case, simply return false. Do not take the ordering of the clusters in the list provided into account when combining the clusters as the ordering is random. Each cluster is formatted as follows:

Cluster id: id, Cluster Name: Name, Cluster Description: description, Cluster Size: size

**Here is the list of clusters:**

{Cluster id: 0, Cluster Name: Corporate Growth Metrics, Cluster Description: This cluster focuses on the key indicators of a company's growth and development. It includes features such as the age of the company, which provides insight into its stability and longevity, and the hiring rate, which is a direct measure of expansion and workforce growth. These features together provide a comprehensive view of a company's growth trajectory., Cluster Size: 2}

{Cluster id: 1, Cluster Name: Liquidity and Trading Activity, Cluster Description: This cluster primarily focuses on the liquidity of assets and trading activity of stocks. It includes features that measure the liquidity of both book and market assets, using Ortiz-Molina and Phillips Liquidity, which is scaled by total assets or market assets. Trading activity is represented through features such as the Amihud Measure, high-low bid-ask spread, dollar trading volume, and share turnover, which provide insights into the illiquidity, price volatility, and trading volume of stocks. The cluster also includes features that capture the variability in trading volume and share turnover, as well as the frequency of zero trades. These features collectively provide a comprehensive view of the liquidity status and trading behavior of stocks over different time periods, ranging from a month to a year., Cluster Size: 12}

[...]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON output consists of an array of arrays each containing the Cluster ids of clusters to be combine.

**Example output:**
{Recombined cluster id: 0, Cluster ids: [0, 12]}
False
[...]

---

## Part 2 — DAG generation (Appendix C)

Run once per final cluster, then once more to link the clusters. Paper text (§14.4.1): the exploration
prompt is run **10 times per group**, and the union of edges forms the undirected graph.

### C1. Causal exploration — candidate edges, undirected (Fig. C14.1)

**system:**
You are an expert financial analyst that determines the causality relationships between finance, accounting and market features in a dataset.

**user:**
Given a dataset with various features for a machine learning model, your task as an AI is to identify causal relationships among these features. Represent these causal connections in the form of a directed acyclic graph (DAG), with features as nodes and causal relationships as edges. A causal relationship between two features means that a change in one feature causes a change in the other feature.An edge from node A to B signifies that changes in A cause changes in B. Consider all feature pairs, ensuring the DAG is acyclic - do not create cycles. Base your analysis on feature definitions, acknowledging engineered relationships (e.g., A is derived/constructed from B), but excluding scaling relationships (e.g. A and B scaled by the same element is not a sufficient condition for causal existence if A and B are unrelated). Utilize expert domain knowledge in finance, accounting and economics to determine the direction of causal relationships. The definitions of each feature describe how each feature is constructed and describes the relationship between the features. Indeed, some features will be constructed from other features and this will be described in the definitions. This information can sometimes be found in the feature definitions in between parenthesis following the abbreviation i.e. for that is. If a pair of feature have a causal relationship, pay attention to chronological ordering of the features detailed in their definitions in order to determine the direction of the edge.Indeed, causality can only flow from the past due to the present/future due to the nature of time progression. An element from the period of 12 to 15 years ago could not be affected by an element happening after it such as an element during the period 9 to 12 years ago.Aim for high recall in identifying causal relationships, avoiding false positives. Return a list of edges in the graph, each with a justification. You must consider the examples below for thought processes and justifications behind the direction of causal relationshipsThese examples contain domain knowledge with respect to valid causal relationship. Here are the examples:

1. If Feature B represents the 30-day average closing price of a stock and Feature A represents the 90-day average closing price, the causal relationship likely flows from A to B. The 90-day average would encompass the 30-day period, influencing its values.
   [....]

**The features to consider are:**fnl_gr1a, coa_gr1a, cowc_gr1a, dbnetis_at, debt_gr3, nfna_gr1a, nncoa_gr1a, col_gr1a, noa_gr1a, noa_at, netdebt_me, ncol_gr1a, ncoa_gr1a, ppeinv_gr1a, lti_gr1a, lnoa_gr1a, inv_gr1a, netis_at, inv_gr1Do not introduce new feature names in the DAG, and exclude elements from feature definitions in the returned edge list.

**Each feature's definition follows the format:** Feature Name: the feature's name, Feature Definition: the feature's definition.
[...]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains the edge forming the causal DAG with a justification for the direction of the causal relationship.

**Example output:**
The list of edges is as follows:
(taccruals_at, cowc_gr1a)(taccruals_at, seas_16_20na)(taccruals_at, taccruals_ni)(oaccruals_at, taccruals_at)(oaccruals_at, seas_16_20na)(oaccruals_at, oaccruals_ni)(oaccruals_at, cowc_gr1a)
[....]

### C2. Causal inference — give the undirected graph directions (Fig. C14.2)

**system:** (same as C1)

**user:**
Given a list of edges forming an undirected graph:(taccruals_at, cowc_gr1a)(taccruals_at, seas_16_20na)(taccruals_at, taccruals_ni)(oaccruals_at, taccruals_at)(oaccruals_at, seas_16_20na)(oaccruals_at, oaccruals_ni)(oaccruals_at, cowc_gr1a)

Each node in this graph represents a feature from a dataset, and each edge signifies the existence of a causal relationship between the two nodes (features) it connects. The ordering of features in each tuple representing an edge does not indicate the direction, as the graph is undirected.Your task is to transform this undirected graph into a directed acyclic graph (DAG) by determining the direction of all edges. All the edge from the undirected graph have to be include in the DAG you generate. Represent these causal connections as a DAG, with features as nodes and causal relationships as directed edges. Ensure the DAG is acyclic. Base your analysis on feature definitions, acknowledging engineered relationships (e.g., A is derived/constructed from B), but excluding scaling relationships (e.g. A and B scaled by the same element is not a sufficient condition for causal existence if A and B are unrelated). Utilize expert domain knowledge in finance and accounting to determine the direction of causal relationships. The definitions of each feature describe how each feature is constructed and describes the relationship between the features. Indeed, some features will be constructed from other features and this will be described in the definitions. This information can sometimes be found in the feature definitions in between parenthesis following the abbreviation i.e. for that is. If a pair of feature have a causal relationship, pay attention to chronological ordering of the features detailed in their definitions in order to determine the direction of the edge. Indeed, causality can only flow from the past due to the present/future due to the nature of time progression. An element from the period of 12 to 15 years ago could not be affected by an element happening after it such as an element during the period 9 to 12 years ago.Return a list of edges in the graph, each with a justification. You must consider the examples below for thought processes and justifications behind the direction of causal relationships. These examples contain domain knowledge with respect to valid causal relationship. **Here are the examples:**

Consider an arbitrary feature A and feature B being calculated using the value of feature A and another arbitrary factor named C. In this case, feature B is created using feature A which means the causal relationship flows from feature A to feature B. It is crucial to recognize that feature B is created by using the value of feature A, rather than the other way around as feature A is an individual feature and feature B is calculated by aggregating the values of different individual features and factors.

[...]

**The features to consider are:** oaccruals_at, taccruals_at, cowc_gr1a, oaccruals_ni, taccruals_ni, seas_16_20na

And the feature definitions are:
[...]

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains the edge forming the causal DAG with a justification for the direction of the causal relationship.

**Example output:**
Here is the format that every edge will have in the list of edges forming the causal DAG that I will return: (Feature Name A, Feature Name B, Justification for causal relationship flowing from feature A to feature B). Here is the list of edges forming the causal DAG:
(oaccruals_at, taccruals_at, taccruals_at is calculated using the value of oaccruals_at (Operating Accruals) and the yearly change in Net financial assets. Therefore, changes in oaccruals_at will affect the value of taccruals_at.)
[...]

### C3. Causal validation — repeated until the LLM says the DAG is valid (Fig. C14.3)

**system:** (same as C1)

**user:**
Evaluate the causal Directed Acyclic Graph (DAG) generated from the provided undirected graph. Examine each edge against guiding principles, feature definitions, thought process examples, and user-provided justifications.

Types of errors to scrutinize:

1. **Overlooking Valid Causal Relationships:**

- Confirm all plausible causal relationships between features. Refer to feature definitions and provided examples for comprehensive coverage.

2. **Directional Errors in Causal Relationships:**

- Review the directionality of each causal relationship (e.g., edge) in the DAG. Align justifications with examples and reasoning in the user's prompt. Verify consistency with feature definitions, excluding scaling relationships.
- Apply finance and accounting domain knowledge to determine causal direction. - Confirm and verify the direction of the edge is coherent with the chronological ordering in feature definitions. - Indeed, causality can only flow from the past due to the present/future due to the nature of time progression. An element constructed using values from the period starting 15 years ago and ending 12 years ago could not be affected by an element happening after it such as an element constructed using values from the period starting 12 years ago and ending 9 years ago.- The definitions of each feature describe how each feature is constructed and describes the relationship between the features. Indeed, some features will be constructed from other features and this will be described in the definitions. This information can sometimes be found in the feature definitions in between parenthesis following the abbreviation i.e. for that is.

3. **Inclusion of Nonexistent Causal Relationships:**

- Verify each edge for genuine causal connections. Cross-reference with prompts, feature definitions, and examples to ensure accuracy.

**Consider the following examples for thought processes and justifications:**

1. Consider Feature A as the average returns of a stock over the previous year (as1y) and Feature B as the average returns over the previous 2 years (as2y). Both features are constructed using the return values from the previous year, suggesting a potential causal relationship. Applying domain knowledge about finance and stock returns, we recognize that past stock returns can influence future
   [...]
   Missing Causal Relationship: the edge was missed in the initial generation of the DAG.
   Nonexistent Causal Relationship: the edge included represents a causal relationship that doesn't exist.
   Directional Errors in Causal Relationships: the direction of the edge is incorrect. The causal relationship flows in the opposite direction than the current direction of the *(cut off in the figure)*

**Parsing step:** Parse GPT-4's output using the Function Calling to obtain valid JSON formatted output. The JSON formatted output contains a Boolean value mentioning if the DAG is valid or contains corrections with justifications.

**Loop:** "The prompt is applied iteratively until all corrections are made the stopping criterion (output in green) is achieved."

**Stopping output:**
assistant: The causal DAG is valid.

---

## Not shown as a prompt: refutation (§14.4.3, text only)

After validation, each edge is tested on data with do-calculus via DoWhy; edges that are not statistically
significant are removed, with a cross-validation check on a subset of the data ("refutation step").
Lasso is then used as a final feature-selection step when fitting models.
