#ifndef PATH_FINDER_H
#define PATH_FINDER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

const int PATH_POINT{2}, FREE_POINT{0}, OBS_POINT{1};
using Point = std::pair<int, int>;

struct Node {
  int x, y;
  int g, h, f;
  Node* parent;
  Node(int x, int y, int g, Node* parent) : x(x), y(y), g(g), parent(parent) {
    f = g;
  }
  Node(int x, int y, int g, int h, Node* parent)
      : x(x), y(y), g(g), h(h), parent(parent) {
    f = g + h;
  }
  bool operator>(const Node& other) const { return f > other.f; }
};

class PathFinder {
 public:
  PathFinder(const std::vector<std::vector<int>>& map, Point start, Point goal);
  std::vector<Node*> AStar();
  std::vector<Node*> Dijkstra();
  void PrintMap(const std::vector<Node*>& path);

 private:
  int Heuristic(Point node);

 private:
  const std::vector<std::vector<int>>& map;
  Point start;
  Point goal;
  int rows, cols;

  // left, up, right, down
  const std::vector<Point> directions = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
};

#endif  // PATH_FINDER_H
