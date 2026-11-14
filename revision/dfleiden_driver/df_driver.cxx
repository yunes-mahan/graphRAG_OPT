// B2b driver: Sahu's Dynamic Frontier (DF) Leiden (github.com/puzzlef/leiden-communities-openmp-dynamic @ 28c304f,
// unmodified headers in ../external/leiden-dynamic/inc) on OUR update streams instead of random batches.
// Input (stdin):  "n T" then, for t = 0..T-1, a line "m_t" followed by m_t undirected edges "u v".
// Version 0: leidenStaticOmp on the initial graph.  Version t>0: apply the batch (insertions only, both directions,
// tidied as in the authors' harness) and run leidenDynamicFrontierOmp from the PREVIOUS DF result's membership,
// vertex weights and community weights (the authors' harness passes the weights of a static Louvain run instead;
// we use the algorithm's own previous state, which is what the method assumes).
// Output (stdout): T lines, the membership of all n vertex ids (vertices not yet present keep their own id).
// Default LeidenOptions (repeat 1, resolution 1, tolerance 1e-2, aggregation tolerance 0.8, max 20 it, 10 passes).
#include <cstdint>
#include <cstdio>
#include <vector>
#include <tuple>
#include <iostream>
#include "inc/main.hxx"
using namespace std;

int main() {
  using K = uint32_t; using V = float;
  omp_set_num_threads(1);                       // single thread: deterministic, like all other maintainers
  size_t n, T; if (scanf("%zu %zu", &n, &T) != 2) return 1;
  DiGraph<K, None, V> y; y.respan(n);
  vector<K> pm; vector<LEIDEN_WEIGHT_TYPE> pv, pc;           // previous DF state (membership, vertex/community weights)
  for (size_t t = 0; t < T; ++t) {
    size_t m; if (scanf("%zu", &m) != 1) return 1;
    vector<tuple<K, K, V>> ins, del; ins.reserve(2 * m);
    for (size_t i = 0; i < m; ++i) {
      unsigned u, v; if (scanf("%u %u", &u, &v) != 2) return 1;
      if (u == v) continue;
      ins.push_back(make_tuple(K(u), K(v), V(1))); ins.push_back(make_tuple(K(v), K(u), V(1)));
    }
    tidyBatchUpdateU(del, ins, y);
    applyBatchUpdateOmpU(y, del, ins);
    LeidenResult<K> r = (t == 0) ? leidenStaticOmp(y, {1})
                                 : leidenDynamicFrontierOmp(y, del, ins, pm, pv, pc, {1});
    for (size_t u = 0; u < n; ++u) printf(u + 1 < n ? "%u " : "%u\n", u < r.membership.size() ? r.membership[u] : K(u));
    fprintf(stderr, "version %zu: |E|=%zu time %.1f ms passes %d\n", t, y.size(), r.time, r.passes);
    pm = r.membership; pv.assign(r.vertexWeight.begin(), r.vertexWeight.end()); pc.assign(r.communityWeight.begin(), r.communityWeight.end());
  }
  return 0;
}
