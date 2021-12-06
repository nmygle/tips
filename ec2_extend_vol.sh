# ファイルの拡張

# パーティションの確認
lsblk
# 拡張
sudo growpart /dev/nvme0n1 1 
# 展開
sudo resize2fs /dev/nvme0n1p1
