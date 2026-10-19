"""
Score predicted graphs the ADIA way: each variable except X and Y gets 1 of 8 roles
from its direct edges with X and Y (official bit rule); score = balanced accuracy.
Adjacency matrices have rows = parents. Run directly to self-check on y_test_reduced.
"""

import networkx as nx
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROLES = {
    # bits: (X -> v) (Y -> v) (v -> Y) (v -> X)
    0b0000: 'Independent',
    0b0001: 'Cause of X',
    0b0010: 'Cause of Y',
    0b0011: 'Confounder',
    0b0100: 'Consequence of Y',
    0b1000: 'Consequence of X',
    0b1010: 'Mediator',
    0b1100: 'Collider',
}


def roles_from_adjacency(adj):
    """{variable: role} for every variable other than X and Y."""
    graph = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)
    if ('X', 'Y') not in graph.edges or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("graph must be a DAG containing X -> Y")
    a = adj.astype(int)
    return {v: ROLES[(a.loc['X', v] << 3) | (a.loc['Y', v] << 2) | (a.loc[v, 'Y'] << 1) | a.loc[v, 'X']]
            for v in adj.columns if v not in ('X', 'Y')}


def role_table(true_graphs, pred_graphs):
    """One row per (dataset, variable): true role and predicted role."""
    missing = set(true_graphs) - set(pred_graphs)
    if missing:
        raise ValueError(f"{len(missing)} datasets have no prediction")
    rows = []
    for dataset_id, true_adj in true_graphs.items():
        true_roles = roles_from_adjacency(true_adj)
        pred_roles = roles_from_adjacency(pred_graphs[dataset_id])
        rows += [{'dataset_id': dataset_id, 'variable': v, 'true': r, 'pred': pred_roles[v]}
                 for v, r in true_roles.items()]
    return pd.DataFrame(rows)


def score(roles_df):
    """Balanced accuracy, plus recall and count for each role."""
    overall = balanced_accuracy_score(roles_df['true'], roles_df['pred'])
    per_role = (roles_df.assign(correct=roles_df['true'] == roles_df['pred'])
                .groupby('true')['correct'].agg(recall='mean', n='size'))
    return overall, per_role


if __name__ == '__main__':
    from paths import ADIA

    truth = pd.read_pickle(ADIA / 'y_test_reduced.pickle')

    perfect, _ = score(role_table(truth, truth))
    print(f"truth vs truth: {perfect:.4f}  (should be 1.0000)")

    def only_x_to_y(adj):
        blank = pd.DataFrame(0, index=adj.index, columns=adj.columns)
        blank.loc['X', 'Y'] = 1
        return blank

    all_indep, per_role = score(role_table(truth, {k: only_x_to_y(a) for k, a in truth.items()}))
    print(f"everything 'Independent': {all_indep:.4f}  (should be 0.1250)")
    print(per_role)