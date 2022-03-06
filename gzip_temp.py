import gzip
import tempfile

import trimesh

source_file = 'temp.stl.gz'

with tempfile.NamedTemporaryFile("w+b", dir="/tmp/", suffix=".stl") as gtemp:
    with gzip.open(source_file, mode="rb") as gfp:
        gtemp.write(gfp.read())
    mesh = trimesh.load(gtemp.name)
