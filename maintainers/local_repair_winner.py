"""Causal, core-anchored community maintenance for a growing graph.

Only the current graph, the inherited assignment, and state accumulated from earlier
calls are used. IDs are never recycled: ordinary local repair changes membership
without renaming communities, while split/merge routines choose the continuing ID
from anchored overlap.
"""

from collections import Counter, defaultdict, deque

import igraph as ig
import leidenalg as la
import numpy as np


class AnchoredBudgetedRepair:
    # Compact controls for the quality/churn trade-off. Gains below are scaled by m,
    # the number of edges, so these thresholds remain interpretable on graph growth.
    MIN_MOVABLE_FRACTION = 0.003
    MAX_MOVABLE_FRACTION = 0.012
    OLD_NODE_MOVE_PENALTY = 0.75
    MOVE_HYSTERESIS = 0.05
    ANCHOR_SUPPORT = 0.68
    MAX_LOCAL_PASSES = 3

    # Structural operations deliberately have separate, substantially higher gates.
    SPLIT_HYSTERESIS = 60.0
    MERGE_HYSTERESIS = 250.0
    MIN_SPLIT_CHILD = 6
    MAX_STRUCTURAL_COMMUNITY = 6000
    MAX_MERGED_COMMUNITY = 10000

    def __init__(self, seed=0):
        self.seed = seed
        self.adj = defaultdict(list)
        self.label = {}
        self.next_id = 0
        self.edge_count = 0

    def _add_edges(self, edges):
        for a, b in edges.T.tolist():
            a, b = int(a), int(b)
            self.adj[a].append(b)
            self.adj[b].append(a)
            self.edge_count += 1

    def initialize(self, nodes, edges, features, years):
        del features, years
        self._add_edges(edges)
        node_list = [int(v) for v in nodes.tolist()]
        index = {v: i for i, v in enumerate(node_list)}
        graph = ig.Graph(
            n=len(node_list),
            edges=[(index[int(a)], index[int(b)]) for a, b in edges.T.tolist()],
            directed=False,
        )
        partition = la.find_partition(
            graph,
            la.ModularityVertexPartition,
            seed=self.seed,
            n_iterations=4,
        )
        for node, community in zip(node_list, partition.membership):
            self.label[node] = int(community)
        self.next_id = max(partition.membership, default=-1) + 1
        return np.asarray([self.label[v] for v in node_list], dtype=np.int64)

    def _community_degrees(self, nodes):
        totals = defaultdict(int)
        for v in nodes:
            totals[self.label[v]] += len(self.adj[v])
        return totals

    def _place_new_nodes(self, new_nodes, all_nodes):
        """Greedy positive modularity-gain voting, including new-node chains."""
        pending = set(new_nodes)
        totals = self._community_degrees(v for v in all_nodes if v not in pending)
        denom = float(max(2 * self.edge_count, 1))

        # Succeeding rounds let a chain inherit support from nodes placed one round
        # earlier. Ordering and all tie breaks are stable.
        for _ in range(4):
            placed = []
            for v in sorted(pending):
                votes = Counter(self.label[u] for u in self.adj[v] if u in self.label)
                if not votes:
                    continue
                degree = len(self.adj[v])
                choices = []
                for community, links in votes.items():
                    gain = links - degree * totals[community] / denom
                    choices.append((gain, links, -community, community))
                gain, _links, _neg_id, community = max(choices)
                if gain > 0.0:
                    self.label[v] = community
                    totals[community] += degree
                    placed.append(v)
            for v in placed:
                pending.remove(v)
            if not placed:
                break

        # Unsupported nodes start genuine fresh lineages; IDs are monotone and are
        # not recycled even if a later move empties a singleton.
        for v in sorted(pending):
            community = self.next_id
            self.next_id += 1
            self.label[v] = community
            totals[community] += len(self.adj[v])

    def _node_stress(self, v):
        degree = len(self.adj[v])
        if degree == 0:
            return 1.0, 0.0
        community = self.label[v]
        internal = sum(1 for u in self.adj[v] if self.label.get(u) == community)
        support = internal / degree
        external_fraction = 1.0 - support
        support_deficit = max(0.0, 0.60 - support)
        disconnected = 1.0 if internal == 0 else 0.0
        stress = external_fraction + 0.80 * support_deficit + 0.50 * disconnected
        return stress, support

    def _ranked_frontier(self, active, old_nodes, node_count):
        scored = []
        stress_by_node = {}
        anchors = set()
        for v in sorted(active):
            stress, support = self._node_stress(v)
            stress_by_node[v] = stress
            if v in old_nodes and len(self.adj[v]) >= 3 and support >= self.ANCHOR_SUPPORT:
                anchors.add(v)
            else:
                scored.append((stress, v))

        aggregate_stress = sum(s for s, _ in scored) / max(len(scored), 1)
        movable_fraction = min(
            self.MAX_MOVABLE_FRACTION,
            self.MIN_MOVABLE_FRACTION
            + (self.MAX_MOVABLE_FRACTION - self.MIN_MOVABLE_FRACTION)
            * min(aggregate_stress, 1.0),
        )
        budget = max(32, int(node_count * movable_fraction))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        selected = [v for _stress, v in ranked[:budget]]

        # One-hop expansion resolves boundary effects, but has its own hard cap.
        selected_set = set(selected)
        expansion = set()
        for v in selected:
            for u in self.adj[v]:
                if u not in anchors and u not in selected_set:
                    expansion.add(u)
        expanded = []
        for u in expansion:
            stress = stress_by_node.get(u)
            if stress is None:
                stress, support = self._node_stress(u)
                if u in old_nodes and len(self.adj[u]) >= 3 and support >= self.ANCHOR_SUPPORT:
                    anchors.add(u)
                    continue
                stress_by_node[u] = stress
            expanded.append((stress, u))
        expanded.sort(key=lambda item: (-item[0], item[1]))
        frontier = selected + [u for _stress, u in expanded[:budget]]
        frontier = sorted(set(frontier), key=lambda v: (-stress_by_node[v], v))
        return frontier, anchors, stress_by_node

    def _local_moves(self, frontier, old_nodes, totals, stress_by_node):
        affected = set()
        moved = set()
        denom = float(max(2 * self.edge_count, 1))
        for _pass in range(self.MAX_LOCAL_PASSES):
            changed = False
            for v in sorted(frontier, key=lambda x: (-stress_by_node[x], x)):
                source = self.label[v]
                degree = len(self.adj[v])
                if degree == 0:
                    continue
                links = Counter(self.label[u] for u in self.adj[v])
                source_links = links.get(source, 0)
                best_gain = float("-inf")
                best_target = source
                for target in sorted(links):
                    if target == source:
                        continue
                    # Exact modularity delta multiplied by m. The source total in
                    # the second term excludes v, while the target total does not.
                    gain = (
                        links[target]
                        - source_links
                        - degree * (totals[target] - totals[source] + degree) / denom
                    )
                    if gain > best_gain or (gain == best_gain and target < best_target):
                        best_gain, best_target = gain, target
                penalty = self.OLD_NODE_MOVE_PENALTY if v in old_nodes else 0.0
                if best_target != source and best_gain > penalty + self.MOVE_HYSTERESIS:
                    self.label[v] = best_target
                    totals[source] -= degree
                    totals[best_target] += degree
                    affected.update((source, best_target))
                    moved.add(v)
                    changed = True
            if not changed:
                break
        return affected, moved

    def _components(self, members, community):
        remaining = set(members)
        components = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            queue = deque([start])
            component = []
            while queue:
                v = queue.popleft()
                component.append(v)
                for u in self.adj[v]:
                    if u in remaining and self.label[u] == community:
                        remaining.remove(u)
                        queue.append(u)
            components.append(component)
        return components

    def _split_incoherent(self, affected, all_nodes, anchors, prior_label, totals):
        if not affected:
            return set()
        members = defaultdict(list)
        for v in all_nodes:
            if self.label[v] in affected:
                members[self.label[v]].append(v)
        created = set()
        denom = float(max(2 * self.edge_count, 1))
        for community in sorted(affected):
            group = members.get(community, [])
            if not (self.MIN_SPLIT_CHILD * 2 <= len(group) <= self.MAX_STRUCTURAL_COMMUNITY):
                continue
            components = self._components(group, community)
            if len(components) <= 1:
                continue

            def lineage_key(component):
                overlap = sum(
                    1 for v in component
                    if v in anchors and prior_label.get(v) == community
                )
                return overlap, len(component), -min(component)

            keeper = max(components, key=lineage_key)
            parent_total = totals[community]
            for component in sorted(
                (c for c in components if c is not keeper), key=lambda c: min(c)
            ):
                if len(component) < self.MIN_SPLIT_CHILD:
                    continue
                component_total = sum(len(self.adj[v]) for v in component)
                gain = component_total * (parent_total - component_total) / denom
                if gain <= self.SPLIT_HYSTERESIS:
                    continue
                child_id = self.next_id
                self.next_id += 1
                for v in component:
                    self.label[v] = child_id
                totals[community] -= component_total
                totals[child_id] = component_total
                parent_total -= component_total
                created.add(child_id)
        return created

    def _merge_adjacent(self, affected, all_nodes, anchors, prior_label, totals):
        if not affected:
            return set()
        sizes = Counter(self.label[v] for v in all_nodes)
        cross_edges = Counter()
        # Exact edge counts for pairs touching a repaired community. v<u ensures
        # each undirected edge is counted once.
        for v in all_nodes:
            source = self.label[v]
            for u in self.adj[v]:
                if v >= u:
                    continue
                target = self.label[u]
                if source != target and (source in affected or target in affected):
                    pair = (source, target) if source < target else (target, source)
                    cross_edges[pair] += 1

        denom = float(max(2 * self.edge_count, 1))
        candidates = []
        for (left, right), links in cross_edges.items():
            if sizes[left] + sizes[right] > self.MAX_MERGED_COMMUNITY:
                continue
            gain = links - totals[left] * totals[right] / denom
            if gain > self.MERGE_HYSTERESIS:
                candidates.append((gain, left, right))
        if not candidates:
            return set()

        # At most one high-confidence merge per version avoids split/merge waves.
        _gain, left, right = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        anchor_mass = Counter()
        for v in anchors:
            current = self.label[v]
            if prior_label.get(v) == current:
                anchor_mass[current] += 1
        keep = max((left, right), key=lambda c: (anchor_mass[c], sizes[c], -c))
        retire = right if keep == left else left
        for v in all_nodes:
            if self.label[v] == retire:
                self.label[v] = keep
        totals[keep] += totals[retire]
        totals.pop(retire, None)
        return {keep, retire}

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        del new_features, new_years
        new_list = [int(v) for v in new_nodes.tolist()]
        all_list = [int(v) for v in all_nodes.tolist()]
        new_set = set(new_list)
        old_nodes = set(all_list) - new_set
        prior_label = {v: self.label[v] for v in old_nodes}

        self._add_edges(new_edges)
        self._place_new_nodes(new_list, all_list)

        # Only nodes touched by graph growth seed stress detection. The selected
        # high-stress set is expanded by one hop under a second equal-size cap.
        active = set(new_list)
        for a, b in new_edges.T.tolist():
            active.add(int(a))
            active.add(int(b))
        frontier, anchors, stress = self._ranked_frontier(active, old_nodes, len(all_list))
        totals = self._community_degrees(all_list)
        affected, _moved = self._local_moves(frontier, old_nodes, totals, stress)

        created = self._split_incoherent(
            affected, all_list, anchors, prior_label, totals
        )
        structural = affected | created
        self._merge_adjacent(structural, all_list, anchors, prior_label, totals)
        return np.asarray([self.label[v] for v in all_list], dtype=np.int64)


def make_maintainer():
    return AnchoredBudgetedRepair(seed=0)
