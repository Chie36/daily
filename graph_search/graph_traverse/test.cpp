#include <iostream>

#include "graph_traverse.h"

void Test1() {
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
}

void Test2() {
  /*
  0 - 1 - 3
  |   |
  2 - 4
  |
  5
  */

  WeightGraph wg(6);
  wg.AddEdge(0, 1, 1);
  wg.AddEdge(0, 2, 4);
  wg.AddEdge(1, 3, 1);
  wg.AddEdge(1, 4, 1);
  wg.AddEdge(2, 4, 1);
  wg.AddEdge(2, 5, 3);
  wg.PrintGraph();

  wg.Dijkstra(0, 5);

  std::vector<int> heuristic = {0, 2, 1, 4, 3, 0};
  wg.AStar(0, 5, heuristic);
}
int main() {
  Test1();
  Test2();
  return 0;
}
