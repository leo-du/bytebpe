# bytebpe
GPT2 style byte-level performant BPE learner and tokenizer

Python binding implemented with PyBind11

## Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```
Then, in the the `build` directory, you can
```python
>>> import bytebpe
>>> bpe = bytebpe.ByteBPE()
>>> bpe.[tab][tab]
bpe.decode(          bpe.encode_token(    bpe.load_from_file(  
bpe.encode_line(     bpe.learn(           bpe.save_to_file(
```
