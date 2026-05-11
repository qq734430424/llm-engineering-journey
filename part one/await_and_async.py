import asyncio
import time

import aiohttp

# async def say_hello():
#     print("Hello")
#     await asyncio.sleep(1)
#     print("World")

# async def main():
#     #await say_hello()
#     #task = asyncio.create_task(say_hello())
#     #await task
#     future = asyncio.ensure_future(say_hello())
#     await future

# asyncio.run(main())



def fetch_url(url):
    """模拟一个耗时的网络请求（同步版本）"""
    print(f"开始获取: {url}")
    time.sleep(2)  # 模拟 2 秒网络延迟
    print(f"完成获取: {url}")
    return f"来自 {url} 的数据"

def main_sync():
    urls = ["http://example.com/1", "http://example.com/2", "http://example.com/3"]
    results = []
    start_time = time.time()

    for url in urls:
        data = fetch_url(url)
        results.append(data)


    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"结果: {results}")


main_sync()

async def fetch_url_async(session, url):
    """模拟一个耗时的网络请求（异步版本）"""
    print(f"开始异步获取: {url}")
    # 注意：这里我们使用 aiohttp 的异步 get 方法，并用 await 等待
    async with session.get(url) as response:
        # 模拟处理响应也需要时间
        await asyncio.sleep(2)  # 使用 asyncio.sleep 模拟 I/O 等待，它不会阻塞线程
        text = await response.text()
        print(f"完成异步获取: {url}")
        return f"来自 {url} 的数据 (长度: {len(text)})"

async def main_async():
    urls = ['https://httpbin.org/get', 'https://httpbin.org/delay/1', 'https://httpbin.org/headers']

    async with aiohttp.ClientSession() as session:  # 创建异步HTTP 会话
        # 为每个 URL 创建一个任务（Task）
        tasks = []
        for url in urls:
            # create_task 会将协程加入事件循环，立即开始调度
            task = asyncio.create_task(fetch_url_async(session,url))
            tasks.append(task)
        
        print("所有任务已创建，等待完成...")

        # 使用 asyncio.gather 并发运行所有任务，并等待它们全部完成
        # gather 返回一个结果列表，顺序与传入的任务顺序一致
        results = await asyncio.gather(*tasks)

        return results
    
start_time = time.time()
final_results = asyncio.run(main_async())
end_time = time.time()
print(f"总耗时: {end_time - start_time:.2f} 秒")
for result in final_results:
    print(result)