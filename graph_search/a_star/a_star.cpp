#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// test
#include <thread>

const int PATH_POINT{8}, FREE_POINT{0}, OBS_POINT{1};
using Point = std::pair<int, int>;
struct Node {
  int x, y;
  double g, h, f;
  Node* parent;
  bool operator<(const Node& other) const { return f < other.f; }
  bool operator==(const Node& other) const {
    return x == other.x && y == other.y;
  }
};

double Heuristic(Node* a, Node* b) {
  return std::abs(a->x - b->x) + std::abs(a->y - b->y);
}

/*
A* 算法的流程：
1 初始化 open_list 和 closed_list
  - open_list: 存储待评估的节点
  - closed_list: 存储已经评估过的节点
2 将起点加入 open_list
3 从 open_list 中选择f(n)值最小的节点作为当前节点
4 如果当前节点是目标节点，则算法结束，返回路径
5 否则，扩展当前节点的邻居节点，并对邻居节点处理：
  - 如果邻居在地图外or占据障碍，则跳过
  - 如果邻居已经在 closed_list 中，则跳过
  - 否则，计算邻居的代价
  - 判断是否需要把邻居加入 open_list
    - 如果邻居已经在 open_list 中，判断是否需要更新其代价
    - 否则，将邻居加入 open_list
6 将当前节点从 open_list 移到 closed_list
7 如果 open_list 空，则没有路径可达目标
*/

std::vector<Node*> AStar(Point start, Point goal,
                         const std::vector<std::vector<int>>& grid) {
  int rows = grid.size();
  int cols = grid[0].size();

  // 1 初始化 open_list 和 closed_list
  // std::priority_queue<Node*, std::vector<Node*>, std::greater<Node*>>
  // open_list;
  std::set<Node*> open_list;
  std::set<Point> closed_list;

  // 2 将起点加入 open_list
  Node* start_node = new Node{.x = start.first,
                              .y = start.second,
                              .g = 0,
                              .h = 0,
                              .f = 0,
                              .parent = nullptr};
  Node* goal_node = new Node{.x = goal.first,
                             .y = goal.second,
                             .g = 0,
                             .h = 0,
                             .f = 0,
                             .parent = nullptr};
  start_node->h = Heuristic(start_node, goal_node);
  start_node->f = start_node->g + start_node->h;
  open_list.insert(start_node);

  while (!open_list.empty()) {
    // 3 从 open_list 中选择f(n)值最小的节点作为当前节点
    Node* current_node = *open_list.begin();
    open_list.erase(open_list.begin());

    std::cout << "current node: " << current_node->x << ", " << current_node->y
              << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 4 如果当前节点是目标节点，则算法结束，返回路径
    if (current_node->x == goal_node->x && current_node->y == goal_node->y) {
      std::vector<Node*> path;
      while (current_node != nullptr) {
        path.push_back(current_node);
        current_node = current_node->parent;
      }
      std::reverse(path.begin(), path.end());
      delete goal_node;
      return path;
    }

    // 5 否则，扩展当前节点的邻居节点，并计算每个邻居的f(n)值：
    std::vector<std::pair<int, int>> neighbors = {
        {-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    for (auto& dir : neighbors) {
      int nx = current_node->x + dir.first;
      int ny = current_node->y + dir.second;

      // 5.1 如果邻居在地图外or占据障碍，则跳过
      if (nx < 0 || nx >= rows || ny < 0 || ny >= cols ||
          grid[nx][ny] == OBS_POINT) {
        continue;
      }

      // 5.2 如果邻居已经在 closed_list 中，则跳过
      if (closed_list.count(Point{nx, ny}) > 0) {
        continue;
      }

      // 5.3 否则，计算邻居的代价
      Node* neighbor_node = new Node{nx, ny, 0, 0, 0, current_node};
      neighbor_node->g = current_node->g + 1;
      neighbor_node->h = Heuristic(neighbor_node, goal_node);
      neighbor_node->f = neighbor_node->g + neighbor_node->h;

      // 5.4 判断是否需要把邻居加入 open_list
      if (false) {
        // 如果邻居已经在 open_list 中，判断是否需要更新其代价

      } else {
        // 否则，将邻居加入 open_list
        open_list.insert(neighbor_node);
      }
    }

    // 6 将当前节点从 open_list 移到 closed_list
    closed_list.insert(Point{current_node->x, current_node->y});
    delete current_node;
  }

  // 7 如果 open_list 空，则没有路径可达目标
  return {};
}

void PrintMap(const std::vector<std::vector<int>>& grid, std::string name) {
  std::cout << " --- " << name << " --- " << std::endl;
  for (const auto& row : grid) {
    for (const auto& ele : row) {
      std::cout << "  " << ele;
    }
    std::cout << std::endl;
  }
};

void PrintMapWithPath(const std::vector<std::vector<int>>& grid,
                      const std::vector<Node*>& path) {
  std::vector<std::vector<int>> grid_copy = grid;
  for (auto n : path) {
    grid_copy[n->x][n->y] = PATH_POINT;
  }
  PrintMap(grid_copy, "grid with path");
}

int main() {
  std::vector<std::vector<int>> grid = {
      {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}};
  PrintMap(grid, "grid");

  Point start{0, 0};
  Point goal{4, 5};

  std::vector<Node*> path = AStar(start, goal, grid);

  if (!path.empty()) {
    std::cout << "Path found:" << std::endl;
    PrintMapWithPath(grid, path);
  } else {
    std::cout << "No path found!" << std::endl;
  }

  return 0;
}
