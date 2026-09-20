#!/bin/bash
# Four-stage layer-pipeline chain, 40 layers split 10 per node.
#
# STATUS: validated, four stages, on all four nodes.
#
# Measured 2026-09-20, one binary on all four nodes (md5 f0f7c5f2d3610fe4de60af7610ae2316), 128
# tokens at temperature 0, --expert-cache-slots 40 named on every stage:
#
#   run 1  6.206 tok/s     run 2  6.091     run 3  6.133     mean 6.143
#
# taken from stage 1's period, which is the chain rate: the embedder cannot start position p+1
# until the sampler's token for p comes back, so its decode loop measures the whole round trip.
# The later stages report higher (6.371 / 6.566 / 6.753) because their clocks start later.
# Slower stages would each report slower, but the chain rate is stage 1's.
#
# Stage 4's output is the model's: " Paris, a city renowned for its rich history, culture, and
# iconic landmarks. Situated in the north-central part of the country, along the Seine River,".
# Stage 1 prints garbage and that is CORRECT - a ten-layer embedder cannot produce tokens, and the
# harness prints whatever its partial forward decodes to.
#
# THE ONE THING THAT MADE THIS WORK: every cross-node connect uses an IP, never a hostname.
# With hostnames, connect() fails with NSPOSIXErrorDomain Code=22 "Invalid argument" and the stage
# gives up after 450 attempts - which is what the first three attempts at this chain hit. The
# addresses are N1 192.168.18.27, N2 192.168.18.25, N3 192.168.18.29, N4 192.168.18.26.
#
# Wiring derived from sources/TinyTitanCLI/PipelineWiring.swift. The roles come from the project
# record: node3 source (embedder, first), node1/node2 both (middles, BACK_SWAP=1), node4 sink
# (sampler, last). node4 cannot originate outbound, so it only ever listens.
#
# Forward edge (increasing layers):  node3 -> node1 -> node2 -> node4
# Back edge (the sampled token):     node4 -> node2 -> node1 -> node3
#
# Every back hop's RECEIVER dials the sender with BACK_SWAP=1, so the token is read from the
# connection that was dialled; node4, which cannot dial, accepts and writes.
set -u
MODEL="$HOME/Downloads/qwen36-4bit.gturbo"
BIN="$HOME/tt-bins/TinyTitanCLI"
LOCAL_BIN="/Users/node4/Downloads/TinyTitan Datacenter/.build/release/TinyTitanCLI"
N1=192.168.18.27
N2=192.168.18.25
N3=192.168.18.29
N4=192.168.18.26
TOKENS=128
CACHE=40   # explicit, so the number is reproducible: the budget defect meant the default used to be wrong

kill_all() {
  pkill -f TinyTitanCLI 2>/dev/null
  for n in node1 node2 node3; do ssh $n@$n 'pkill -f TinyTitanCLI' 2>/dev/null; done
}
kill_all; sleep 2

# --- node4: last stage, sampler. Listens on both edges, never dials. ---
env TINYTITAN_STAGE_LISTEN=47721 \
    TINYTITAN_STAGE_BACK_LISTEN=47712 \
    TINYTITAN_STAGE_BACK_ROLE=sink \
    "$LOCAL_BIN" --model "$MODEL" --prompt "The capital of France is" \
    --max-new "$TOKENS" --temperature 0 --layer-range 30:40 \
    --expert-cache-slots "$CACHE" > /tmp/c4.out 2> /tmp/c4.err &
sleep 3

# --- node2: middle, 20:30 ---
ssh node2@node2 "env TINYTITAN_STAGE_LISTEN=47742 \
  TINYTITAN_STAGE_CONNECT=$N4:47721 \
  TINYTITAN_STAGE_BACK_LISTEN=47720 \
  TINYTITAN_STAGE_BACK_CONNECT=$N4:47712 \
  TINYTITAN_STAGE_BACK_SWAP=1 \
  TINYTITAN_STAGE_BACK_ROLE=both \
  nohup ~/tt-bins/TinyTitanCLI --model ~/Downloads/qwen36-4bit.gturbo --prompt 'The capital of France is' \
  --max-new $TOKENS --temperature 0 --layer-range 20:30 \
  --expert-cache-slots $CACHE > /tmp/c2.out 2> /tmp/c2.err" &
sleep 3

# --- node1: middle, 10:20 ---
ssh node1@node1 "env TINYTITAN_STAGE_LISTEN=47741 \
  TINYTITAN_STAGE_CONNECT=$N2:47742 \
  TINYTITAN_STAGE_BACK_LISTEN=47730 \
  TINYTITAN_STAGE_BACK_CONNECT=$N2:47720 \
  TINYTITAN_STAGE_BACK_SWAP=1 \
  TINYTITAN_STAGE_BACK_ROLE=both \
  nohup ~/tt-bins/TinyTitanCLI --model ~/Downloads/qwen36-4bit.gturbo --prompt 'The capital of France is' \
  --max-new $TOKENS --temperature 0 --layer-range 10:20 \
  --expert-cache-slots $CACHE > /tmp/c1.out 2> /tmp/c1.err" &
sleep 3

# --- node3: first stage, embedder; drives the run ---
START=$(python3 -c 'import time;print(time.time())')
ssh node3@node3 "env TINYTITAN_STAGE_CONNECT=$N1:47741 \
  TINYTITAN_STAGE_BACK_CONNECT=$N1:47730 \
  TINYTITAN_STAGE_BACK_SWAP=1 \
  TINYTITAN_STAGE_BACK_ROLE=source \
  timeout 900 ~/tt-bins/TinyTitanCLI --model ~/Downloads/qwen36-4bit.gturbo --prompt 'The capital of France is' \
  --max-new $TOKENS --temperature 0 --layer-range 0:10 \
  --expert-cache-slots $CACHE > /tmp/c3.out 2> /tmp/c3.err; echo \"node3 exit=\$?\""
END=$(python3 -c 'import time;print(time.time())')
echo "wall for the whole chain: $(python3 -c "print(f'{$END-$START:.1f}s')")"

sleep 2
echo "=== node3 (stage 1, layers 0:10) — its period IS the chain rate ==="
ssh node3@node3 'grep -E "stop=|tok/s|error" /tmp/c3.err | tail -3'
echo "  output: $(ssh node3@node3 'head -c 150 /tmp/c3.out')"
for n in 1 2; do
  echo "=== node$n ==="
  ssh node$n@node$n "grep -E 'stop=|error' /tmp/c$n.err | tail -2"
done
echo "=== node4 (stage 4, layers 30:40) ==="
grep -E "stop=|error" /tmp/c4.err | tail -2
echo "  output: $(head -c 150 /tmp/c4.out)"

kill_all
echo "=== chain stopped ==="
