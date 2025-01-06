# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.ui import webui_main

if __name__ == '__main__':
    app, local_url, share_url = webui_main()
    print(f'local_url: {local_url}')
    print(f'share_url: {share_url}')