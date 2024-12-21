
#include "graph_traverse.h"

void Graph::AddEdge(int u, int v) {
  adj_list[u].push_back(v);
  adj_list[v].push_back(u);
}

void Graph::PrintGraph() const {
  std::cout << "Graph (Adjacency List):" << std::endl;
  for (int i = 0; i < v; ++i) {
    std::cout << "Vertex " << i << " :";
    for (size_t j = 0; j < adj_list[i].size(); ++j) {
      std::cout << " " << adj_list[i][j];
      if (j < adj_list[i].size() - 1) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

void Graph::BFS(int start) const {
  // 创数组v和队列q
  std::vector<bool> visited(v, false);
  std::queue<int> q;

  // 加起点入队列q，且加起点入v
  q.push(start);
  visited[start] = true;

  std::cout << "BFS starting from vertex " << start << ": ";

  while (!q.empty()) {
    // 弹出q的第一个点
    int node = q.front();
    q.pop();

    // 输出当前点
    std::cout << node << " ";

    // 扩当前点邻居
    for (int neighbor : adj_list[node]) {
      // 检邻居是否不在v里
      if (!visited[neighbor]) {
        // 加邻居入队列q，且加邻居入v
        q.push(neighbor);
        visited[neighbor] = true;
      }
    }
  }
  std::cout << std::endl;
}

void Graph::DFS(int start) const {
  // 创数组v和栈s
  std::vector<bool> visited(v, false);
  std::stack<int> s;

  // 加起点入栈s，且加起点入v
  s.push(start);
  visited[start] = true;

  std::cout << "DFS starting from vertex " << start << ": ";

  while (!s.empty()) {
    // 弹出s顶的点
    int node = s.top();
    s.pop();

    // 输出当前点
    std::cout << node << " ";

    // 扩当前点邻居（逆序添加，确保按递增顺序访问）
    for (int i = adj_list[node].size() - 1; i >= 0; --i) {
      int neighbor = adj_list[node][i];
      // 检邻居是否不在v里
      if (!visited[neighbor]) {
        // 加邻居入栈s，且加邻居入v
        s.push(neighbor);
        visited[neighbor] = true;
      }
    }
  }
  std::cout << std::endl;
}

void Graph::DFS2(int start) const {
  // 创数组v
  std::vector<bool> visited(v, false);

  std::function<void(int)> Recursive = [&](int node) {
    // 加当前点入v
    visited[node] = true;

    // 输出当前点
    std::cout << node << " ";

    // 扩当前点邻居（
    for (int neighbor : adj_list[node]) {
      // 检邻居是否不在v里
      if (!visited[neighbor]) {
        // 邻居入递归函数
        Recursive(neighbor);
      }
    }
  };

  std::cout << "DFS2 starting from vertex " << start << ": ";
  // 起点入递归函数
  Recursive(start);
  std::cout << std::endl;
}

void WeightGraph::AddEdge(int u, int v, int weight) {
  adj_list[u].push_back({v, weight});
  adj_list[v].push_back({u, weight});
}

void WeightGraph::PrintGraph() const {
  std::cout << "WeightGraph (Adjacency List):" << std::endl;
  for (int i = 0; i < v; ++i) {
    std::cout << "Vertex " << i << " :";
    for (size_t j = 0; j < adj_list[i].size(); ++j) {
      std::cout << " " << adj_list[i][j].first << "(" << adj_list[i][j].second
                << ")";
      if (j < adj_list[i].size() - 1) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

void WeightGraph::Dijkstra(int start, int goal) const {
  // 准备数组g, parent
  std::vector<int> g(v, std::numeric_limits<int>::max());
  g[start] = 0;
  std::vector<int> parent(v, -1);

  // 创优先队列open_list
  using Node = std::pair<int, int>;  // {g(n), node}
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;

  // 将起点加入open_list
  open_list.push({g[start], start});

  std::cout << "Dijkstra Search from vertex " << start << " to vertex " << goal
            << ": ";

  while (!open_list.empty()) {
    // 从open_list中选择g值最小的节点作为当前节点
    int current = open_list.top().second;
    open_list.pop();

    // 如果当前节点是目标节点，则算法结束，输出路径
    if (current == goal) {
      std::vector<int> path;
      while (current != -1) {
        path.push_back(current);
        current = parent[current];
      }
      std::reverse(path.begin(), path.end());
      for (int node : path) {
        std::cout << node << " ";
      }
      std::cout << std::endl;
      return;
    }

    // 否则，扩展当前节点的邻居
    for (const auto& pair : adj_list[current]) {
      int node = pair.first;
      int weight = pair.second;

      // 如果邻居节点未被访or找到更优路径，则添加邻居节点到open_list
      int tentative_g = g[current] + weight;
      if (tentative_g < g[node]) {
        parent[node] = current;
        g[node] = tentative_g;
        open_list.push({g[node], node});
      }
    }
  }

  std::cout << "No path found" << std::endl;
}

void WeightGraph::AStar(int start, int goal,
                        const std::vector<int>& heuristic) const {
  // 准备数组g, f, parent
  std::vector<int> g(v, std::numeric_limits<int>::max());
  g[start] = 0;
  std::vector<int> f(v, std::numeric_limits<int>::max());
  f[start] = g[start] + heuristic[start];
  std::vector<int> parent(v, -1);

  // 创优先队列open_list
  using Node = std::pair<int, int>;  // {f(n), node}
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;

  // 将起点加入open_list
  open_list.push({f[start], start});

  std::cout << "A* Search from vertex " << start << " to vertex " << goal
            << ": ";

  while (!open_list.empty()) {
    // 从open_list中选择f值最小的节点作为当前节点
    int current = open_list.top().second;
    open_list.pop();

    // 如果当前节点是目标节点，则算法结束，输出路径
    if (current == goal) {
      std::vector<int> path;
      while (current != -1) {
        path.push_back(current);
        current = parent[current];
      }
      std::reverse(path.begin(), path.end());
      for (int node : path) {
        std::cout << node << " ";
      }
      std::cout << std::endl;
      return;
    }

    // 否则，扩展当前节点的邻居
    for (const auto& pair : adj_list[current]) {
      int node = pair.first;
      int weight = pair.second;

      // 如果邻居节点未被访or找到更优路径，则添加邻居节点到open_list
      int tentative_g = g[current] + weight;
      if (tentative_g < g[node]) {
        parent[node] = current;
        g[node] = tentative_g;
        f[node] = g[node] + heuristic[node];
        open_list.push({f[node], node});
      }
    }
  }

  std::cout << "No path found" << std::endl;
}
