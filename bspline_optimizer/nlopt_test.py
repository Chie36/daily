import nlopt
import numpy as np


# 定义目标函数以及梯度
def objective_function(x, grad):
    if grad is not None:
        grad[0] = 2 * (x[0] - 3)  # 计算 x_0 的梯度
        grad[1] = 2 * (x[1] - 2)  # 计算 x_1 的梯度
    cost = (x[0] - 3) ** 2 + (x[1] - 2) ** 2  # 计算目标函数值
    print(f"当前迭代: {x}, 目标代价值: {cost}")
    return cost


# 创建优化器对象，选择 LD_LBFGS 算法，指定优化维度为 2
opt = nlopt.opt(nlopt.LD_LBFGS, 2)

# 设置目标函数
opt.set_min_objective(objective_function)

# 设置最大迭代次数
opt.set_maxeval(100)

# 设置停止容忍度
opt.set_xtol_rel(1e-4)

# 初始猜测
initial_guess = [0, 0]

# 执行优化
try:
    result = opt.optimize(initial_guess)
    print(f"优化结果: {result}")
except Exception as e:
    print(f"优化失败: {e}")
