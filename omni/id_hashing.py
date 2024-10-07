import hashlib

def string_to_hash_positions(input_string):
    # Step 1: Hash the input string using SHA-256
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[:1]


import random
seen_strings = {}
def get_random_unseen_string():
    l = random.randint(1, 10)
    chars = 'abcdefghijklmnopqrstuvwxyz_0123456789'
    while True:
        s = "spkr_"
        for i in range(l):
            c = random.choice(chars)
            s += c
        if s not in seen_strings:
            break
    
    return s


seen = set()

collisions = 0
total = 0
for i in range(1000000):
    s = get_random_unseen_string()
    
    hash = string_to_hash_positions(s)
    if s in seen:
        collisions += 1    
    total += 1
    seen.add(hash)

print(collisions, total)