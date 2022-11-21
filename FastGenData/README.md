# FastGenData

Faster GenData, roughly use the same algorithm as [GenData](../GenData/draft3.py) but with higher throughput.

Throughput around 20-30 image/second on my machine (6 cpu core).

The main improvement come from porting to Rust and
exploiting parallelism.

## Build&Run

1. Install Rust toolchain (eg. via [rustup](`https://rustup.rs/`))
2. Type `cargo build --release` in terminal to compile.
   - Executable location: `./target/release/FastGenData`
3. Run `./target/release/FastGenData <n>` where n is amount of image to generate.
4. The output by default are in `results` folder
