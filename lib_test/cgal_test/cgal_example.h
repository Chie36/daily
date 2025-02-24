#ifndef CGAL_EXAMPLE_H
#define CGAL_EXAMPLE_H

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/intersections.h>

#include <iostream>
#include <list>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef K::Segment_2 Segment_2;
typedef std::list<Point_2> Polygon;

class GeometryOperations {
 public:
  // 计算最小矩形包围盒
  static void ComputeBoundingBox(const std::vector<Point_2>& points);

  // 计算两个线段的交点
  static void ComputeIntersection(const Segment_2& seg1, const Segment_2& seg2);

  // 计算三角形的面积
  static double ComputeArea(const Point_2& p1, const Point_2& p2,
                            const Point_2& p3);

  // 计算两点之间的欧几里得距离
  static double ComputeDistance(const Point_2& p1, const Point_2& p2);

  // 计算凸包
  static void ComputeConvexHull(const std::vector<Point_2>& points);
};

#endif  // CGAL_EXAMPLE_H
