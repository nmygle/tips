# https://qiita.com/pokari_dz/items/0f14a21e3ca3df025d21
# ファイル名に「○○」という文字列が含まれているファイルのリストを取得。
find [検索対象フォルダのパス] -type f -name "*[検索したい文字列]*"

# ファイルを開いた中身に、「○○」という文字列が含まれているファイルのリスト。
grep [検索したい文字列] -rl [検索対象フォルダのパス]
