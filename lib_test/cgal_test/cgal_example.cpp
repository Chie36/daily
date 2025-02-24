#include "cgal_example.h"

// 计算最小矩形包围盒
void GeometryOperations::ComputeBoundingBox(
    const std::vector<Point_2>& points) {
  CGAL::Bbox_2 bbox = CGAL::bbox_2(points.begin(), points.end());
  std::cout << "Bounding Box: [" << bbox.xmin() << ", " << bbox.ymin()
            << "] -> [" << bbox.xmax() << ", " << bbox.ymax() << "]"
            << std::endl;
}

// 计算两个线段的交点
void GeometryOperations::ComputeIntersection(const Segment_2& seg1,
                                             const Segment_2& seg2) {
  auto result = CGAL::intersection(seg1, seg2);
  if (result) {
    if (const Point_2* p = boost::get<Point_2>(&*result)) {
      std::cout << "Intersection point: (" << p->x() << ", " << p->y() << ")"
                << std::endl;
    } else {
      std::cout << "The segments are collinear and overlap." << std::endl;
    }
  } else {
    std::cout << "The segments do not intersect." << std::endl;
  }
}

// 计算三角形的面积
double GeometryOperations::ComputeArea(const Point_2& p1, const Point_2& p2,
                                       const Point_2& p3) {
  return CGAL::area(p1, p2, p3);
}

// 计算两点之间的欧几里得距离
double GeometryOperations::ComputeDistance(const Point_2& p1,
                                           const Point_2& p2) {
  return CGAL::sqrt(CGAL::squared_distance(p1, p2));
}

// 计算凸包
void GeometryOperations::ComputeConvexHull(const std::vector<Point_2>& points) {
  Polygon convex_hull;
  CGAL::convex_hull_2(points.begin(), points.end(),
                      std::back_inserter(convex_hull));

  std::cout << "Convex Hull Points:" << std::endl;
  for (const auto& point : convex_hull) {
    std::cout << "(" << point.x() << ", " << point.y() << ")" << std::endl;
  }
}