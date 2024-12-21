
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