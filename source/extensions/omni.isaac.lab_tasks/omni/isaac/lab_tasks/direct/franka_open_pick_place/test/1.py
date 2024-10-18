from gymnasium import spaces

a = spaces.Box(low=-1.0, high=1.0, shape=(8,))
print(a.low, a.high, a.shape)
