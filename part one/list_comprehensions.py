# list comprehensions（列表推导式）是一种简洁的创建列表的方式。它允许我们在一行代码中生成一个新的列表，通常比使用传统的 for 循环更简洁和易读。
# 下面是一些使用列表推导式的示例：
# 1. 创建一个包含前10个自然数的列表
squares = [x**2 for x in range(10)]
print(squares)  # 输出: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 2. 从一个列表中筛选出偶数
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)  # 输出: [2, 4, 6, 8, 10]

# 3. 将一个列表中的字符串转换为大写
fruits = ['apple', 'banana', 'cherry']
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(uppercase_fruits)  # 输出: ['APPLE', 'BANANA', 'CHERRY']

# 4. 创建一个包含元组的列表，每个元组包含一个数及其平方
squares_tuples = [(x, x**2) for x in range(10)]
print(squares_tuples)  # 输出: [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49), (8, 64), (9, 81)]

# 5. 创建一个元组的推导式
squares_set = {x**2 for x in range(10)}
print(squares_set)  # 输出: {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# 6. 创建一个set的推导式
squares_dict = {x: x**2 for x in range(10)}
print(squares_dict)  # 输出: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}