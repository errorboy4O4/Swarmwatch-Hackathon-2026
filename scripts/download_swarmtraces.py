"""Download the public Swarm Traces redacted dataset (~15 MB) into data/swarmtraces/."""
import os, urllib.request
URL = "https://swarmtraces.org/data/final/redacted.jsonl.gz"
os.makedirs("data/swarmtraces", exist_ok=True)
dst = "data/swarmtraces/redacted.jsonl.gz"
urllib.request.urlretrieve(URL, dst)
print(dst, os.path.getsize(dst), "bytes")
