# Part I: 表格解决法
  在本书的第一部分，我们以简洁的方式阐述了强化学习算法的核心思想：当状态空间和行动空间足够小时，其价值函数的近似可以通过数组或表格来表示。在此条件下，我们可以找到最优价值函数和最优策略。这与本书第二部分中的近似方法形成了鲜明对比，近似方法只寻找近似解，在解决大型问题时更加高效。

  第一章聚焦于单一状态强化学习问题的解决方法，即老虎机问题 **（译者注：原文为bandit problem，直译为海盗问题，但在此上下文中，翻译为老虎机问题更为贴切）** 。第二章则介绍了本书处理一般问题所采用的框架——有限马尔可夫决策过程，及其核心理念，包括Bellman方程和值函数。

  接下来的三章分别详细阐述了解决有限马尔可夫决策问题的三种基本方法：动态规划、蒙特卡洛和时序差分学习。每种方法都有其独特的优势与局限：动态规划方法在数学上严谨且成熟，但其应用需依赖于对环境完整准确的建模；蒙特卡洛方法无需模型，概念直观，却不适合进行逐步增量计算；时序差分方法则兼具无需模型与完全增量的特点，但其分析过程较为复杂。这三种方法的效率和收敛速度各有不同。

  最后两章则深入探讨了如何将上述三种方法有机融合，以获得每类方法的最佳特性。第六章通过采用多步自举的策略，将蒙特卡洛方法与时序差分方法的优势相结合。第七章分析了如何将时序差分学习方法与模型学习及规划方法（如动态规划）无缝衔接，为表格型强化学习问题提供一个全面、统一的解决策略。
