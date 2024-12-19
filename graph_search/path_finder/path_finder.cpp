#include "path_finder.h"

PathFinder::PathFinder(const std::vector<std::vector<int>>& map, Point start,
                       Point goal)
    : map(map),
      start(start),
      goal(goal),
      rows(map.size()),
      cols(map[0].size()) {}

void PathFinder::PrintMap(const std::vector<Node*>& path) {
  std::vector<std::vector<int>> map_copy = map;
  for (auto node : path) {
    if (node != nullptr) {
      map_copy[node->x][node->y] = PATH_POINT;
    }
  }

  for (int i = 0; i < map.size(); ++i) {
    for (int j = 0; j < map[0].size(); ++j) {
      if (map_copy[i][j] == FREE_POINT) {
        std::cout << ". ";
      } else if (map_copy[i][j] == OBS_POINT) {
        std::cout << "# ";
      } else if (map_copy[i][j] == PATH_POINT) {
        std::cout << "o ";
      }
    }
    std::cout << std::endl;
  }
}

int PathFinder::Heuristic(Point node) {
  return std::abs(node.first - goal.first) +
         std::abs(node.second - goal.second);
}

std::vector<Node*> PathFinder::AStar() {
  // 1. 初始化open_list和closed_list
  std::priority_queue<Node*, std::vector<Node*>, std::greater<Node*>>
      open_list_pq;
  std::unordered_map<int, std::unordered_map<int, Node*>> open_list;
  std::unordered_map<int, std::unordered_map<int, bool>> closed_list;

  auto AddNodeToOpenList = [&](Node* node) {
    open_list_pq.push(node);
    open_list[node->x][node->y] = node;
  };

  // 2. 将起点加入open_list
  Node* start_node =
      new Node{start.first, start.second, 0, Heuristic(start), nullptr};
  AddNodeToOpenList(start_node);

  while (!open_list_pq.empty()) {
    // 3. 从open_list中选择f值最小的节点作为当前节点
    Node* current_node = open_list_pq.top();
    open_list_pq.pop();

    // 4. 如果当前节点是目标节点，则算法结束，返回路径
    if (current_node->x == goal.first && current_node->y == goal.second) {
      std::vector<Node*> path;
      while (current_node != nullptr) {
        path.push_back(current_node);
        current_node = current_node->parent;
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    // 5. 否则，扩展当前节点的邻居
    for (auto dir : directions) {
      int new_x = current_node->x + dir.first;
      int new_y = current_node->y + dir.second;

      // 5.1 如果邻居在地图外、占据障碍、在closed_list中，则跳过
      if (new_x < 0 || new_y < 0 || new_x >= rows || new_y >= cols) continue;
      if (map[new_x][new_y] == OBS_POINT) continue;
      if (closed_list[new_x][new_y]) continue;

      // 5.2 否则，新建邻居节点
      int new_g = current_node->g + 1;
      int new_h = Heuristic({new_x, new_y});
      Node* neighbor_node = new Node(new_x, new_y, new_g, new_h, current_node);

      // 5.3 如果邻居节点未被访or找到更优路径，则添加邻居节点到open_list
      if (open_list[new_x][new_y] == nullptr ||
          neighbor_node->f < open_list[new_x][new_y]->f) {
        AddNodeToOpenList(neighbor_node);
      }
    }

    // 6. 将当前节点从open_list移到closed_list
    closed_list[current_node->x][current_node->y] = true;
  }
  return {};
}

std::vector<Node*> PathFinder::Dijkstra() {
  // 1. 初始化open_list和closed_list
  std::priority_queue<Node*, std::vector<Node*>, std::greater<Node*>>
      open_list_pq;
  std::unordered_map<int, std::unordered_map<int, Node*>> open_list;
  std::unordered_map<int, std::unordered_map<int, bool>> closed_list;

  auto AddNodeToOpenList = [&](Node* node) {
    open_list_pq.push(node);
    open_list[node->x][node->y] = node;
  };

  // 2. 将起点加入open_list
  Node* start_node = new Node{start.first, start.second, 0, nullptr};
  AddNodeToOpenList(start_node);

  while (!open_list_pq.empty()) {
    // 3. 从open_list中选择g值最小的节点作为当前节点
    Node* current_node = open_list_pq.top();
    open_list_pq.pop();

    // 4. 如果当前节点是目标节点，则算法结束，返回路径
    if (current_node->x == goal.first && current_node->y == goal.second) {
      std::vector<Node*> path;
      while (current_node != nullptr) {
        path.push_back(current_node);
        current_node = current_node->parent;
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    // 5. 否则，扩展当前节点的邻居
    for (auto dir : directions) {
      int new_x = current_node->x + dir.first;
      int new_y = current_node->y + dir.second;

      // 5.1 如果邻居在地图外、占据障碍、在closed_list中，则跳过
      if (new_x < 0 || new_y < 0 || new_x >= rows || new_y >= cols) continue;
      if (map[new_x][new_y] == OBS_POINT) continue;
      if (closed_list[new_x][new_y]) continue;

      // 5.2 否则，新建邻居节点
      int new_g = current_node->g + 1;
      Node* neighbor_node = new Node(new_x, new_y, new_g, current_node);

      // 5.3 如果邻居节点未被访or找到更优路径，则添加邻居节点到open_list
      if (open_list[new_x][new_y] == nullptr ||
          neighbor_node->g < open_list[new_x][new_y]->g) {
        AddNodeToOpenList(neighbor_node);
      }
    }

    // 6. 将当前节点从open_list移到closed_list
    closed_list[current_node->x][current_node->y] = true;
  }
  return {};
}
