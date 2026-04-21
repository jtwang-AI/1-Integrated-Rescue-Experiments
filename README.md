
集成救援实验 (Integrated Rescue Experiments)
目录包含用于以下模块的工程代码： 

GMRACCR.py：具有冲突或协作角色的角色分配模块。 

Firstorder_ellipse.py：原始的一阶椭圆控制器演示程序。 

integrated_role_formation.py：集成的海上搜救分配与圆周编队模拟器。 

run_integrated_experiments.py：可复现实验的入口程序。 

代码功能说明
该集成流水线将角色分配决策转化为运动层面的执行： 
根据无人潜航器（UUV/USV）的能力和救援角色权重构建资格矩阵。 
使用 GMRA 或 GMRACCR 模型求解角色分配问题。 
将分配的角色映射到以目标为中心的圆周轨道半径和相位偏移。 
在以下场景下模拟一阶编队维持情况： 
基线部署（Nominal deployment）。 


稀疏船队多角色部署（Scarce-fleet multi-role deployment）。 


中继失效与在线重新分配（Relay failure and online reassignment）。 


海流干扰与碰撞风险压力测试（Ocean-current disturbance and collision-risk stress）。 

运行方式

输出结果将写入以下位置： 

results/summary.json

results/metrics.csv

轨迹图位于 results/*.png 

运行独立求解器演示：

运行原始椭圆控制器演示：

安装 pulp，GMRACCR.py 将针对此处使用的较小案例回退到精确穷举求解器。 

主要结果文件

results/summary.json：嵌套的机器可读实验结果。 

results/metrics.csv：用于电子表格或附录的扁平化结果表。 

results/static_gmraccr.png：优化的名义部署图。 

results/scarce_gmraccr.png：兼容性感知的稀疏船队部署图。 

results/failure_reassign.png：中继失效恢复图。 

results/disturbance_current_only.png：无避障情况下的紧凑启动海流干扰压力测试图。 

results/disturbance_collision_aware.png：带有避障控制的相同压力测试图。 

与论文的对应关系

名义部署：集成分配 + 编队维持。 

稀疏船队部署：多角色负载下的 GMRA 与 GMRACCR 对比。 

失效响应实验：丢失中继后的重新分配。 
