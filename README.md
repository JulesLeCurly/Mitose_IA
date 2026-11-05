# Mitose_IA

## Where is the code?

That's a great question. Honestly, we've been asking ourselves the same thing.

There *was* code. It lived right here, a perfectly ordinary Python script. Then it trained a neural network to memorize itself, byte by byte, checked that the copy was perfect... and deleted itself. What you're looking at is the aftermath: the code is still here, it's just not stored as text anymore. It's stored as **weights**.

So no, you can't read the source. You can only ask the model nicely to remember it.

## Why store code like this?

Efficiency, obviously. The original script weighed about 5 KB. The model that remembers it weighs about 280 KB, more than 50 times bigger. It takes 72,200 parameters and several minutes of training to remember about 5,000 characters, and getting them back is a lot slower than, say, opening a text file. A compression ratio of roughly 0.02. Revolutionary. Investors, please form an orderly queue.

Bonus feature: nobody knows *how* the code is stored in there. Not you, not us, not even the model. Open `Mitose_IA.npz` and all you'll find is 72,200 floating-point numbers. Somewhere in that soup, the network came up with its own private representation of the code, and it's not sharing. You can't read it, you can't `grep` it, you can't fix a typo in it. You can only ask.

## The idea

A program that goes through *mitosis*: it copies itself into a neural network, then the original disappears and only the copy remains.

The network is basically a lookup table in disguise: you give it a position in the file, it tells you which character is there.

### What goes in: a position, written in binary

The input is the position of a character in the file, written as 16 bits (zeros and ones). We start counting at 0:

| Character         | Position | Input of the network |
|-------------------|----------|----------------------|
| the first one     | 0        | `0000000000000000`   |
| the second one    | 1        | `0000000000000001`   |
| the third one     | 2        | `0000000000000010`   |
| ...               | ...      | ...                  |
| the 101st one     | 100      | `0000000001100100`   |

That's just the number 100 converted to binary, padded with zeros up to 16 digits. With 16 bits we can count up to 65,535, which is plenty for a small script.

### What comes out: a character, also in binary

The output is 8 numbers. Each one is the network's vote for one bit: above 0 means `1`, below 0 means `0`. Those 8 bits form one byte, a number between 0 and 255.

And a byte is how a computer stores a character. Every character has a number, agreed upon once and for all by an encoding (ASCII, extended by UTF-8). For example:

| Character | Number | 8 bits     |
|-----------|--------|------------|
| `A`       | 65     | `01000001` |
| `a`       | 97     | `01100001` |
| `"`       | 34     | `00100010` |
| space     | 32     | `00100000` |

So when we ask the network about position `0000000000000000`, it should answer `00100010`, which is `"`: the first character of the script (it starts with a docstring). Ask about every position in turn, glue the characters together, and you get the whole file back.

One last trick: how does the network say "the file is over"? It answers `00000000` (the byte `0x00`), a character that never appears in real source code.

The script overfits a small MLP (16 → 256 → 256 → 8, about 72k parameters) on its own source code until every single bit comes out right, with a safety margin. It then saves the weights to `Mitose_IA.npz`, regenerates the code from that file using only numpy, and checks that the result is byte-for-byte identical to the original. If it is, the source can be deleted.

Generalization? Never heard of it. This is the one task where overfitting is the goal.

## Getting the code back

Put `Mitose_IA.npz` in a folder and run this (numpy only, no torch needed). It asks the network for the file one byte at a time: "what's at position 0? position 1? position 2?..." until the network answers with the end marker.

```python
import numpy as np

w = np.load("Mitose_IA.npz")
code = b""
for position in range(2 ** 16):
    # 1. Write the position in binary: 5 -> [0, 0, ..., 1, 0, 1]
    x = np.array([int(bit) for bit in format(position, "016b")], dtype=np.float32)
    # 2. Pass it through the 3 layers of the network
    x = np.maximum(x @ w["w0"] + w["b0"], 0)
    x = np.maximum(x @ w["w1"] + w["b1"], 0)
    x = x @ w["w2"] + w["b2"]
    # 3. Each output above 0 is a 1: the 8 outputs give one byte
    byte = int("".join("1" if value > 0 else "0" for value in x), 2)
    if byte == 0:  # end of file
        break
    code += bytes([byte])

open("Mitose_IA.py", "wb").write(code)
```

Congratulations, you've just brought the program back to life. Yes, technically those lines are code. We don't talk about it.

Don't let step 2 scare you: those three slightly cryptic lines are just the **forward pass** of a small neural network (an MLP), and that's the only "AI knowledge" you need here. The `@` is numpy's matrix multiplication: `x @ w["w0"]` mixes the 16 input bits into 256 new numbers, `+ w["b0"]` adds a little offset (the bias) to each of them, and `np.maximum(..., 0)` replaces every negative number with 0 (that's the ReLU, the network's way of saying "meh, ignore that"). Do it three times (skipping the clipping the last time, since we want to see the negative votes) and you go from 16 bits to 256 numbers, to 256 numbers again, and finally to 8. Multiply, add, clip, repeat. That's the whole magic trick; all the intelligence is in the weights.

## Running the mitosis yourself

Requirements: Python 3, `numpy`, `torch`.

```bash
python Mitose_IA.py
```

It prints the training progress, then `MITOSIS SUCCESSFUL` if the saved model reproduces the script exactly. Then the self-destruct countdown starts: 10 seconds to read the terminal and say goodbye, and the script deletes itself. Changed your mind? Press `Ctrl+C` during the countdown, or set `DELETE_SOURCE = False` at the top of the file before running it.

Note: the model memorizes the exact file it was trained on. Change a single character in the script and you need to train again.
