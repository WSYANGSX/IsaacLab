from gymnasium import spaces

a = spaces.Box(low=-1, high=1, shape=(7,))
print(a.low)
