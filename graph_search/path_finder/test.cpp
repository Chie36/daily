#include <iostream>

#include "path_finder.h"

int main() {
  std::vector<std::vector<int>> map = {{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}};
  Point start = {0, 0};
  Point goal = {11, 11};
  PathFinder path_finder(map, start, goal);

  std::cout << "Original Map:" << std::endl;
  path_finder.PrintMap({});

  std::vector<Node*> path_a_star = path_finder.AStar();
  if (!path_a_star.empty()) {
    std::cout << "\nA Star Path Found!" << std::endl;
    path_finder.PrintMap(path_a_star);
  } else {
    std::cout << "\nNo path found!" << std::endl;
  }

  std::vector<Node*> path_dijkstra = path_finder.Dijkstra();
  if (!path_dijkstra.empty()) {
    std::cout << "\nDijkstra Path Found!" << std::endl;
    path_finder.PrintMap(path_dijkstra);
  } else {
    std::cout << "\nNo path found!" << std::endl;
  }

  return 0;
}
