"""
装饰器 = 函数包装函数 + 不修改原代码扩展功能
@ 语法本质是函数替换
wrapper 才是真正执行的函数
推荐使用 *args, **kwargs 提高通用性
支持函数、类、甚至带参数的装饰器
"""


def decorator_function(original_function):
    def wrapper_function(*args, **kwargs):
        print("Wrapper executed this before {}".format(original_function.__name__))
        return original_function(*args, **kwargs)
    return wrapper_function

@decorator_function
def display():
    print("Display function ran")

display()

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()


def decorator_with_arguments(func):
    def wrapper(*args, **kwargs):
        print("Arguments were: {} {}".format(args, kwargs))
        return func(*args, **kwargs)
    return wrapper

@decorator_with_arguments
def display_info(name, age):
    print("display_info ran with arguments ({}, {})".format(name, age))

display_info("John", 25)



def repeat(num_times):
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                print("Arguments were: {} {}".format(args, kwargs))
                func(*args, **kwargs)
        return wrapper
    return decorator_repeat

@repeat(num_times=1)
def greet(name):
    print("Hello {}!".format(name))

greet("Alice")



def log_class_decorator(cls):
    class WrappedClass:
        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)

        def __getattr__(self, attr):
            return getattr(self.wrapped, attr)
        
        def display2(self):
            print("Before calling display2")
            result = self.wrapped.display2()
            print("After calling display")
            return result

    return WrappedClass


@log_class_decorator
class MyClass:
    def display2(self):
        print("Display method of MyClass")

obj = MyClass()
obj.display2()