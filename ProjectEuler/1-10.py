#%% Problem 1
nums = []
for i in range(1000):
    if i % 3 == 0 or i % 5 == 0:
        print(i)
        nums.append(i)
print(f'Problem 1 Sum: {sum(nums)}')

#%% Problem 2
a, b = 0, 1

nums = []
while b < 4000000:
  a, b = b, a + b
  if b % 2 == 0:
    print(b)
    nums.append(b)

print(f'Problem 2 Sum: {sum(nums)}')

