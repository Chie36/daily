import numpy as np
import matplotlib.pyplot as plt

class PathAnalysis:
    def __init__(self, points):
        """
        初始化 PathAnalysis 对象。
        :param points: 路径点的列表，每个点是一个元组 (x, y)。
        """
        self.points = points
    
    def calculate_heading_angles(self):
        """
        计算连续路径点之间的航向角。
        :return: 航向角的列表（单位：度）。
        """
        headings = []
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            # 计算方向角
            angle = np.arctan2(y2 - y1, x2 - x1)
            # 转换为航向角
            heading = (np.pi / 2 - angle) * 180 / np.pi
            if heading < 0:
                heading += 360
            headings.append(heading)
        return headings
    
    def calculate_curvature(self):
        """
        使用二阶导数法计算路径的曲率。
        :return: 曲率的列表。
        """
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        
        # 计算 x 和 y 的一阶和二阶导数
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # 使用公式计算曲率
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        
        curvature = numerator / denominator
        return curvature[1:-1]  # 避免边界效应
    
    def analyze(self):
        """
        执行路径分析：计算航向角和曲率。
        :return: 包含航向角和曲率的字典。
        """
        heading_angles = self.calculate_heading_angles()
        curvatures = self.calculate_curvature()
        return {
            'heading_angles': heading_angles,
            'curvatures': curvatures
        }

    def plot_path(self):
        """
        绘制路径、航向角和曲率。
        """
        # 提取路径的 X 和 Y 坐标
        x_points, y_points = zip(*self.points)
        
        plt.figure(figsize=(14, 6))

        # 绘制路径的点和路径线
        plt.subplot(1, 3, 1)
        plt.plot(x_points, y_points, marker='o', color='b', label="Path")
        plt.title("Path of the Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.annotate('Start', (x_points[0], y_points[0]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
        plt.annotate('End', (x_points[-1], y_points[-1]), textcoords="offset points", xytext=(0,10), ha='center', color='red')

        # 绘制航向角
        if len(self.points) > 1:
            heading_angles = self.calculate_heading_angles()
            plt.subplot(1, 3, 2)
            plt.plot(heading_angles, marker='o', color='r', label="Heading Angles")
            plt.title("Heading Angles along the Path")
            plt.xlabel("Point Index")
            plt.ylabel("Heading Angle (degrees)")
            plt.grid(True)
            plt.legend()
            plt.annotate('Heading Angle', (len(heading_angles)//2, heading_angles[len(heading_angles)//2]), textcoords="offset points", xytext=(0,10), ha='center', color='green')

        # 绘制曲率
        if len(self.points) > 2:
            curvatures = self.calculate_curvature()
            plt.subplot(1, 3, 3)
            plt.plot(curvatures, marker='o', color='g', label="Curvatures")
            plt.title("Curvature along the Path")
            plt.xlabel("Point Index")
            plt.ylabel("Curvature")
            plt.grid(True)
            plt.legend()
            plt.annotate('Curvature', (len(curvatures)//2, curvatures[len(curvatures)//2]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')

        plt.tight_layout()
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 生成 50 个沿平滑正弦曲线的路径点
    t = np.linspace(0, 4 * np.pi, 100)
    x_points = t
    y_points = np.sin(t)  # 一个简单的平滑正弦波
    
    # 将 X 和 Y 点组合成路径点的列表
    points = list(zip(x_points, y_points))
    
    # 创建 PathAnalysis 对象
    path_analysis = PathAnalysis(points)
    
    # 执行分析
    result = path_analysis.analyze()
    
    # 打印结果
    print("Heading Angles:")
    print(result['heading_angles'])
    print("Curvatures:")
    print(result['curvatures'])

    # 绘制路径和结果
    path_analysis.plot_path()
