# Python Utility for PGet ⥥
# https://github.com/replicate/pget
# Use this script with a pget manifest file:
# from pget import pget_manifest
# pget_manifest('manifest.pget')

# Your manifest file must be in shape:
# https://example.com/image1.jpg /local/path/to/image1.jpg
# https://example.com/document.pdf /local/path/to/document.pdf
# https://example.com/weights.pth /local/path/to/weights.pth
# ... etc ..

# Read more about pget multifile downloads here:
# https://github.com/replicate/pget?tab=readme-ov-file#multi-file-mode

import os
import subprocess
import time

def pget_manifest(manifest_filename: str='manifest.pget'):
  start = time.time()
  with open(manifest_filename, 'r') as f:
    manifest = f.read()

  to_dl = []
  # ensure directories exist
  for line in manifest.splitlines():
    _, path = line.split(" ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
      to_dl.append(line)

  # write new manifest
  with open("tmp.pget", 'w') as f:
    f.write("\n".join(to_dl))

  # download using pget
  subprocess.check_call(["pget", "multifile", "tmp.pget"])

  # log metrics
  timing = time.time() - start
  print(f"Downloaded weights in {timing} seconds")
