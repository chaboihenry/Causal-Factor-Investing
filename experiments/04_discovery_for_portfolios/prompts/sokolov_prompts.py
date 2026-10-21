"""
Sokolov et al. (2025) DAG prompts C1-C3 as templates, word for word from Appendix C.
Our additions are marked RECONSTRUCTED or ADDED; everything else is their printed text, typos kept.
Only the examples printed in the paper are used; the [...] gaps are left out, not invented.
"""

SYSTEM = ("You are an expert financial analyst that determines the causality relationships between "
          "finance, accounting and market features in a dataset.")

C1 = (
    "Given a dataset with various features for a machine learning model, your task as an AI is to identify causal "
    "relationships among these features. Represent these causal connections in the form of a directed acyclic graph "
    "(DAG), with features as nodes and causal relationships as edges. A causal relationship between two features means "
    "that a change in one feature causes a change in the other feature.An edge from node A to B signifies that changes "
    "in A cause changes in B. Consider all feature pairs, ensuring the DAG is acyclic - do not create cycles. Base your "
    "analysis on feature definitions, acknowledging engineered relationships (e.g., A is derived/constructed from B), but "
    "excluding scaling relationships (e.g. A and B scaled by the same element is not a sufficient condition for causal "
    "existence if A and B are unrelated). Utilize expert domain knowledge in finance, accounting and economics to "
    "determine the direction of causal relationships. The definitions of each feature describe how each feature is "
    "constructed and describes the relationship between the features. Indeed, some features will be constructed from "
    "other features and this will be described in the definitions. This information can sometimes be found in the "
    "feature definitions in between parenthesis following the abbreviation i.e. for that is. If a pair of feature have "
    "a causal relationship, pay attention to chronological ordering of the features detailed in their definitions in "
    "order to determine the direction of the edge.Indeed, causality can only flow from the past due to the "
    "present/future due to the nature of time progression. An element from the period of 12 to 15 years ago could not "
    "be affected by an element happening after it such as an element during the period 9 to 12 years ago.Aim for high "
    "recall in identifying causal relationships, avoiding false positives. Return a list of edges in the graph, each "
    "with a justification. You must consider the examples below for thought processes and justifications behind the "
    "direction of causal relationshipsThese examples contain domain knowledge with respect to valid causal "
    "relationship. Here are the examples:\n\n"
    "1. If Feature B represents the 30-day average closing price of a stock and Feature A represents the 90-day "
    "average closing price, the causal relationship likely flows from A to B. The 90-day average would encompass the "
    "30-day period, influencing its values.\n\n"
    "The features to consider are:{names}\n\n"   # RECONSTRUCTED: line break; the figure shows the last name glued to "Do"
    "Do not introduce new feature names in the DAG, and exclude elements from "
    "feature definitions in the returned edge list.\n\n"
    "Each feature's definition follows the format: Feature Name: the feature's name, Feature Definition: the feature's "
    "definition.\n{definitions}"
)

C2 = (
    "Given a list of edges forming an undirected graph:{edges}\n\n"
    "Each node in this graph represents a feature from a dataset, and each edge signifies the existence of a causal "
    "relationship between the two nodes (features) it connects. The ordering of features in each tuple representing an "
    "edge does not indicate the direction, as the graph is undirected.Your task is to transform this undirected graph "
    "into a directed acyclic graph (DAG) by determining the direction of all edges. All the edge from the undirected "
    "graph have to be include in the DAG you generate. Represent these causal connections as a DAG, with features as "
    "nodes and causal relationships as directed edges. Ensure the DAG is acyclic. Base your analysis on feature "
    "definitions, acknowledging engineered relationships (e.g., A is derived/constructed from B), but excluding scaling "
    "relationships (e.g. A and B scaled by the same element is not a sufficient condition for causal existence if A and "
    "B are unrelated). Utilize expert domain knowledge in finance and accounting to determine the direction of causal "
    "relationships. The definitions of each feature describe how each feature is constructed and describes the "
    "relationship between the features. Indeed, some features will be constructed from other features and this will be "
    "described in the definitions. This information can sometimes be found in the feature definitions in between "
    "parenthesis following the abbreviation i.e. for that is. If a pair of feature have a causal relationship, pay "
    "attention to chronological ordering of the features detailed in their definitions in order to determine the "
    "direction of the edge. Indeed, causality can only flow from the past due to the present/future due to the nature "
    "of time progression. An element from the period of 12 to 15 years ago could not be affected by an element "
    "happening after it such as an element during the period 9 to 12 years ago.Return a list of edges in the graph, "
    "each with a justification. You must consider the examples below for thought processes and justifications behind "
    "the direction of causal relationships. These examples contain domain knowledge with respect to valid causal "
    "relationship. Here are the examples:\n\n"
    "Consider an arbitrary feature A and feature B being calculated using the value of feature A and another arbitrary "
    "factor named C. In this case, feature B is created using feature A which means the causal relationship flows from "
    "feature A to feature B. It is crucial to recognize that feature B is created by using the value of feature A, "
    "rather than the other way around as feature A is an individual feature and feature B is calculated by aggregating "
    "the values of different individual features and factors.\n\n"
    "The features to consider are: {names}\n\n"
    "And the feature definitions are:\n{definitions}"
)

C3 = (
    "Evaluate the causal Directed Acyclic Graph (DAG) generated from the provided undirected graph. Examine each edge "
    "against guiding principles, feature definitions, thought process examples, and user-provided justifications.\n\n"
    "Types of errors to scrutinize:\n\n"
    "1. **Overlooking Valid Causal Relationships:**\n"
    "- Confirm all plausible causal relationships between features. Refer to feature definitions and provided examples "
    "for comprehensive coverage.\n"
    "2. **Directional Errors in Causal Relationships:**\n"
    "- Review the directionality of each causal relationship (e.g., edge) in the DAG. Align justifications with "
    "examples and reasoning in the user's prompt. Verify consistency with feature definitions, excluding scaling "
    "relationships.\n"
    "- Apply finance and accounting domain knowledge to determine causal direction. - Confirm and verify the direction "
    "of the edge is coherent with the chronological ordering in feature definitions. - Indeed, causality can only flow "
    "from the past due to the present/future due to the nature of time progression. An element constructed using "
    "values from the period starting 15 years ago and ending 12 years ago could not be affected by an element happening "
    "after it such as an element constructed using values from the period starting 12 years ago and ending 9 years "
    "ago.- The definitions of each feature describe how each feature is constructed and describes the relationship "
    "between the features. Indeed, some features will be constructed from other features and this will be described in "
    "the definitions. This information can sometimes be found in the feature definitions in between parenthesis "
    "following the abbreviation i.e. for that is.\n"
    "3. **Inclusion of Nonexistent Causal Relationships:**\n"
    "- Verify each edge for genuine causal connections. Cross-reference with prompts, feature definitions, and examples "
    "to ensure accuracy.\n"
    "Consider the following examples for thought processes and justifications:\n"
    "1. Consider Feature A as the average returns of a stock over the previous year (as1y) and Feature B as the "
    "average returns over the previous 2 years (as2y). Both features are constructed using the return values from the "
    "previous year, suggesting a potential causal relationship. Applying domain knowledge about finance and stock "
    "returns, we recognize that past stock returns can influence future returns.\n"   # RECONSTRUCTED: "returns." (cut off in figure)
    "Missing Causal Relationship: the edge was missed in the initial generation of the DAG.\n"
    "Nonexistent Causal Relationship: the edge included represents a causal relationship that doesn't exist.\n"
    "Directional Errors in Causal Relationships: the direction of the edge is incorrect. The causal relationship flows "
    "in the opposite direction than the current direction of the edge.\n\n"       # RECONSTRUCTED: "edge." (cut off)
    # ADDED: the current DAG is restated each round, since corrections change it
    "The current DAG is (Feature Name A, Feature Name B, Justification for causal relationship flowing from feature A "
    "to feature B):\n{dag}"
)

# ADDED (replaces OpenAI Function Calling, unavailable to both models): identical for every model
JSON_EDGES = ('\n\nReturn your answer as JSON only, inside a ```json code block, with exactly this structure: '
              '{"edges": [{"source": "feature name", "target": "feature name", "justification": "text"}]}. '
              'An edge from source to target means changes in source cause changes in target.')
JSON_VALIDATE = ('\n\nReturn your answer as JSON only, inside a ```json code block, with exactly this structure: '
                 '{"valid": true or false, "corrections": [{"type": "missing" or "nonexistent" or "direction", '
                 '"source": "feature name", "target": "feature name", "justification": "text"}]}. '
                 'For "missing", give the edge to add. For "nonexistent" and "direction", give the edge exactly as it '
                 'currently appears in the DAG. If the DAG is valid, return {"valid": true, "corrections": []}.')