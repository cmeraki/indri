from glob import iglob
import numpy as np

total = 0
for f in iglob('/home/apurva/.cache/indri/instruct_tokens/*.npy'):
    x = np.load(f)
    total += len(x)

print(total)
