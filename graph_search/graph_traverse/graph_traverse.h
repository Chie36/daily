#ifndef GRAPH_TRAVERSE_H
#define GRAPH_TRAVERSE_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
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

 public:
  int v;
  std::vector<std::vector<int>> adj_list;
};

class WeightGraph : public Graph {
 public:
  WeightGraph(int v) : Graph(v) { adj_list.resize(v); }
  void AddEdge(int u, int v, int weight);
  void PrintGraph() const;
  void Dijkstra(int start, int goal) const;
  void AStar(int start, int goal, const std::vector<int>& heuristic) const;

 public:
  std::vector<std::vector<std::pair<int, int>>> adj_list;
};

#endif  // GRAPH_TRAVERSE_H
