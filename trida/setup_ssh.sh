#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_path>" >&2
    exit 1
fi

input_path="$1"
input_file="${input_path}/deepspeed-hostfile"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '${input_file}' not found" >&2
    exit 1
fi

# 读取输入文件
content=$(cat "$input_file")
echo "Input content: $content" >&2

# 获取当前主机名
current_host=$(hostname -s)
echo "Current host is $current_host" 1>&2

# 逐行读取文件并处理每个节点
while read -r line; do
    # 提取主机名（忽略 slots 部分）
    host=$(echo $line | awk '{print $1}')

    if [ "$host" != "$current_host" ]; then
        echo "Processing host: $host" 1>&2

        # 在当前主机上添加目标主机的密钥
        echo "This is $current_host, adding $host to known hosts..." 1>&2
        ssh-keyscan -H $host >> ~/.ssh/known_hosts
        echo "This is $current_host, known hosts have been updated." 1>&2

        # 显示当前用户的公钥
        echo "Displaying current user's public key:"
        cat ~/.ssh/id_rsa.pub

        # 复制 SSH 密钥到目标主机
        echo "Copying SSH key to chengyuew@$host" 1>&2
        ssh-copy-id "chengyuew@$host"
        echo "Finished copying SSH key to chengyuew@$host" 1>&2

        # 请求目标主机复制其 SSH 密钥到当前主机
        echo "Requesting $host to copy its SSH key to $current_host" 1>&2
        ssh "chengyuew@$host" "ssh-copy-id chengyuew@$current_host"
    fi
done < "$input_file"

echo "SSH key exchange completed" 1>&2

