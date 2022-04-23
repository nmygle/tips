# ジョブを投げ完了していればその結果を受け取り、完了していなければ別のプロセスを動かす

from concurrent.futures import ThreadPoolExecutor
import time

def make_udon(kind):
    print(f'## {kind}うどんを作ります。')
    time.sleep(3)
    return kind + 'うどん'

kinds = ['たぬき', 'かけ']#, 'ざる', 'きつね', '天ぷら', '肉']
executor = ThreadPoolExecutor(max_workers=1)
futures = {}

for kind in kinds:
    print(f'## {kind}うどん オーダー入りました。')
    future = executor.submit(make_udon, kind)
    futures[kind] = future

for t in range(10):
    for kind in futures:
        if futures[kind].done():
            print(f"## {kind}うどん 出来上がりました={futures[kind].result()}")

    print(f"--running({t})--")
    for kind in futures:
        print(f"{kind}: {futures[kind].running()}")
    print(f"--done({t})--")
    for kind in futures:
        print(f"{kind}: {futures[kind].done()}")
    print("--\n")
    if all([futures[kind].done() for kind in futures]):
        break
    time.sleep(1)
    #print('%sお待たせしました。' % future.result())

executor.shutdown()
