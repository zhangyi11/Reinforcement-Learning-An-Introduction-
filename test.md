公式2.3的更新规则是本书中最常出现的更新规则，一般形式为

$$新估计值 \leftarrow 旧估计值+ 步长[目标-旧估计值]$$

表达式$`[目标-旧估计值]`$是估计的误差。通过向‘目标’迈进以减小误差。目标指示了一个理想的前进方向，尽管目标可能带有噪声。在老虎机问题中目标是第n次获得的奖励。

我们注意到，在公式2.3增量方法中使用的步长参数会随着时间步长的变化而变化。处理动作a第n次奖励时，增量方法使用的步长参数为$`\frac{1}{n}`$。本书中我们用$`\alpha`$或$`\alpha_t(a)`$来表示步长参数。

以下伪代码使用了样本平均法，$`\epsilon`$-贪婪方法和增量公式表示老虎机问题的算法，函数bandit(a)返回动作a相应的奖励。
![image](https://github.com/zhangyi11/Reinforcement-Learning-An-Introduction-/blob/main/images/A%20simple%20bandit%20algorithm.jpg)

## 2.5 追踪非平稳问题
到目前为止讨论的求平均值方法都是针对稳态老虎机问题，即奖励不随时间的变化而变化。如前所述，我们接触的大多数强化学习问题都是非稳态的。在非稳态的强化学习问题中，短期内获得奖励的权重高于长期的奖励权重是有意义的。最常用的方法就是使用一个固定的步长参数。例如在增量公式2.3中，用前n-1次奖励更新平均值$`Q_n`$的公式可以调整为

$$Q_{n+1}\dot{=}Q_n+\alpha[R_n-Q_n]$$

其中$`\alpha\in(0,1]`$。公式表明$`Q_{n+1}`$是初始估计值$`Q_1`$和过去奖励和的加权平均。

$$ Q_{n+1} = Q_n+\alpha[R_n-Q_n] = \alpha R_n+(1-\alpha)Q_n = \alpha R_n+(1-\alpha)[\alpha R_{n-1}+(1-\alpha)Q_{n-1}]=(1-\alpha)^nQ_1+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-1}R_i$$

$$ 其中，(1-\alpha)^n + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-1} = 1，即权重之和为1 $$

权重$`\alpha(1-\alpha)^{n-1}`$取决于$`R_i`$之间有多少奖励，$`1-\alpha`$的值小于1，介于两个奖励之间的奖励数量越多，$`R_i`$的权重就越小。实际上，权重根据$`1-\alpha`$上的指数呈指数衰减（如果$`1-\alpha=0`$，那么所有的权重都放在最后的奖励$`R_n`$上，因为$`0^0=1`$）。

$`\alpha_n(a)`$表示用于处理第n次选择行动a获得的奖励参数。如前所述，样本平均法的步长参数$`\alpha_n(a)=\frac{1}{n}`$，大数定律保证了该方法可以收敛到真实的行动价值。但是无法保证序列$`{\alpha_n(a)}`$的所有选择都能收敛。随机近似理论中的著名结论确保了与概率1收敛所需要的条件

$$\sum_{n=1}^{\infty}\alpha_n(a)=\infty \ \ 和\ \  \sum_{n=1}^{\infty}\alpha_n^2(a)<\infty \ \ (2.7)\ $$

第一个条件是必须得，以保证在步数足够大的情况下，可以克服任何初始条件和随机波动。第二个条件确保步长最终会变得足够小，以保证收敛性。

对于样本平均方法论来说,$`\alpha_n(a)=\frac{1}{n}`$满足2.7中的两个条件，但恒定步长参数如$`\alpha_n(a)=\alpha`$则不满足2.7中的第二个条件，这表明估计值不会完全的收敛，而是根据最近收到的奖励变化，如前所述，在非平稳强化学习中恒定的步长参数是最常见的。此外，满足条件(2.7)的步长参数收敛速度一般很慢，或者需要进行大量的调整才能获得令人满意的收敛速度。总的来说，满足2.7收敛条件的步长参数序列在理论工作中经常使用，但在实际中却很少使用。

_练习2.4_ 如果步长参数$`\alpha_n`$不是恒定的，估计价值$`Q_n`$是先前获得奖励的加权平均值，请仿照公式2.6，给出新的公式
_答案_  

$$ Q_{n+1} = Q_n + \alpha_n[R_n-Q_n]=\alpha_nR_n+(1-\alpha_n)Q_n=\alpha_nR_n+(1-\alpha_n)[\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1}]=(1-\alpha_1)^nQ_1+\sum_{i=1}^n\alpha_iR_i\prod_{j=i+1}^n(1-\alpha_j) $$

_练习2.5_ （编程）请设计一个实验，以展现样本平均法在非稳态问题时遇到的困难。例如，使用10臂老虎机，最初老虎机每个臂的$`q_*(a)`$相等，然后每个臂的真实奖励随机独立改变（例如，每一步中，给所有的$`q_*(a)`$添加一个均值为0，方差为0.01的增量。）使用样本平均法、增量公式、以及恒定的步长参数如$`\alpha=0.1`$和$`\epsilon`$-贪婪方法（$`\epsilon`$=0.1），运行10000步。

_答案_
```
import numpy as np

class Ten_Armed_Testbed:
    def __init__(self,num_arms=10,epsilon=0.1,alpha=0.1):
        self.arms = num_arms
        self.epsilon = epsilon
        self.true_values = [0] * num_arms   # 行动的真实价值，初始值为0
        self.estimate_values = [0] *num_arms  # 行动的估计价值
        self.count = [0] * num_arms  # 记录每个行动的选择次数（仅样本平均法，即非固定步长时使用）
        self.alpha = alpha

    def get_action(self):
        if np.random.random() > self.epsilon:
            return np.argmax(self.estimate_values)
        else: return np.random.choice(self.arms)

    def step(self,action,constant_parameter=False):
        reward = np.random.normal(self.true_values[action],1)  # 假定每个行动的实际奖励值服从均值为q*(a)，方差为1的正态分布。
        self.true_values = [i+np.random.normal(0,0.01) for i in self.true_values]  # 每一个行动后，给给所有的q*(a)添加一个均值为0，方差为0.01的增量
        self.count[action] += 1
        if constant_parameter:
            self.estimate_values[action] += self.alpha*(reward - self.estimate_values[action])  # 增量公式，固定步长，计算行动的估计价值
        else:
            if self.count[action] != 0:
                self.estimate_values[action] += 1/self.count[action]*(reward - self.estimate_values[action])



def run_experiment(bandit, steps=100000, trials=10):  # 步数在1w时，样本平均法有时会优于固定参数，但是步数在10w时，固定参数总是优于样本平均法
    for _ in range(trials):
        for constant_step_size in [False, True]:
            for _ in range(steps):
                action = bandit.get_action()
                bandit.step(action, constant_step_size)
            error = np.abs(np.mean(np.array(bandit.true_values) - np.array(bandit.estimate_values)))
            print(f"{constant_step_size = },{error = }")
        print("-"*10)
        # print(f"{bandit.count = }")

if __name__ == "__main__":
    bandit = Ten_Armed_Testbed()
    run_experiment(bandit)

#(译者注：上述代码是译者自己写的，可能存在问题）
```

