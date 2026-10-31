"""Audio helpers: sound as numbers, phone sound, background noise, sound files.

Inside the code, sound is always a float32 numpy array of mono samples in
[-1, 1], plus its sample rate. Bytes (16-bit PCM) only appear at the wire.
"""
