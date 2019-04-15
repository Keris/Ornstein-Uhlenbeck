# Ornstein-Uhlenbeck Process

> 注意：这里的OU流程作用与四轴飞行器控制任务。

它会根据高斯（正态）分布生成随机样本，但是每个样本都会影响到后续样本，使两个连续样本很接近。因此本质上是马尔可夫流程。

为何与我们有关呢？我们可以直接从高斯分布中抽样吧？是的，但是注意，我们希望使用该流程向动作中添加一些噪点，以便促进探索性行为。因为动作就是应用到飞行器上的力和扭矩，我们希望连续动作不要变化太大。否则，我们可能不会飞到任何地方！想象下随机上下左右翻转控制器！

除了样本的暂时相关特性外，OU 流程的另一个好处是在一段时间之后，它会逐渐接近指定的均值。在用它生成噪点时，我们可以指定均值为 0，当我们在学习任务上取得进展时，它将能够减少探索行为。

Ornstein-Uhlenbeck 流程实现：

```python
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```
