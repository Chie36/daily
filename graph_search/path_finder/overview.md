# A* 算法的流程：
- 初始化
1. 初始化open_list和closed_list
2. 将起点加入open_list
- 循环内
3. 从open_list中选择f值最小的节点作为当前节点
4. 如果当前节点是目标节点，则算法结束，返回路径
5. 否则，扩展当前节点的邻居
    - 如果邻居在地图外、占据障碍、在closed_list中，则跳过
    - 否则，新建邻居节点
    - 如果邻居节点未被访or找到更优路径，则添加邻居节点到open_list
6. 将当前节点从open_list移到closed_list
# Dijkstra 算法的流程：
- 初始化
1. 初始化open_list和closed_list
2. 将起点加入open_list
- 循环内
3. 从open_list中选择g值最小的节点作为当前节点
4. 如果当前节点是目标节点，则算法结束，返回路径
5. 否则，扩展当前节点的邻居
    - 如果邻居在地图外、占据障碍、在closed_list中，则跳过
    - 否则，新建邻居节点
    - 如果邻居节点未被访问过or找到更优路径，则添加邻居节点到open_list
6. 将当前节点从open_list移到closed_list
