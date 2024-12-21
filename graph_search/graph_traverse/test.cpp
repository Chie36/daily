#include <iostream>

#include "graph_traverse.h"

int main() {
  /*

    0 - 1 - 3
    |   |
    2   4
    |
    5

  */

  Graph g(6);
  g.AddEdge(0, 1);
  g.AddEdge(0, 2);
  g.AddEdge(1, 3);
  g.AddEdge(1, 4);
  g.AddEdge(2, 5);
  g.PrintGraph();

  g.BFS(0);
  g.DFS(0);
  g.DFS2(0);

  return 0;
}
