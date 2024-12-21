#ifndef GRAPH_TRAVERSE_H
#define GRAPH_TRAVERSE_H

#include <functional>
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

// undirected unweighted graph
class Graph {
 public:
  Graph(int v) : v(v) { adj_list.resize(v); }
  void AddEdge(int u, int v);
  void PrintGraph() const;
  void BFS(int start) const;
  void DFS(int start) const;
  void DFS2(int start) const;

 private:
  int v;
  std::vector<std::vector<int>> adj_list;
};

#endif  // GRAPH_TRAVERSE_H
