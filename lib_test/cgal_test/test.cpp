#include "cgal_example.h"

int main() {
  // 测试: 最小矩形包围盒
  std::vector<Point_2> points1 = {Point_2(1, 1), Point_2(4, 1), Point_2(4, 3),
                                  Point_2(1, 3)};
  GeometryOperations::ComputeBoundingBox(points1);

  // 测试: 计算线段交点
  Segment_2 seg1(Point_2(0, 0), Point_2(4, 4));
  Segment_2 seg2(Point_2(0, 4), Point_2(4, 0));
  GeometryOperations::ComputeIntersection(seg1, seg2);

  // 测试: 计算三角形面积
  Point_2 p1(0, 0), p2(5, 0), p3(5, 5);
  std::cout << "Triangle area: " << GeometryOperations::ComputeArea(p1, p2, p3)
            << std::endl;

  // 测试: 计算两点之间的距离
  Point_2 p4(0, 0), p5(3, 4);
  std::cout << "Distance between points: "
            << GeometryOperations::ComputeDistance(p4, p5) << std::endl;

  // 测试: 计算凸包
  std::vector<Point_2> points2 = {Point_2(0, 0), Point_2(2, 1), Point_2(3, 3),
                                  Point_2(1, 2), Point_2(0, 3)};
  GeometryOperations::ComputeConvexHull(points2);

  return 0;
}
