import os

# 禁用 Xet 下载后端
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download
from huggingface_hub.utils import set_client_factory, httpx

# 绕过 SSL 证书验证，如果有证书，这段代码可以注释掉
set_client_factory(
    lambda: httpx.Client(
        verify=False,
        follow_redirects=True,
        timeout=60.0
    )
)

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="data/bge-m3",
    max_workers=4
)

print("BGE-M3 下载完成！")