#!/bin/bash
# Four-stage layer-pipeline chain, 40 layers split 10 per node.
#
# STATUS: the TWO-STAGE half is validated; the FOUR-STAGE half is not working yet.
#
# Validated 2026-09-20, all four nodes on the same binary (md5 f0f7c5f2d3610fe4de60af7610ae2316):
#   node4: TINYTITAN_STAGE_LISTEN=47721 STAGE_BACK_LISTEN=47712 STAGE_BACK_ROLE=sink \
#          --layer-range 20:40
#   node2: TINYTITAN_STAGE_CONNECT=<node4>:47721 STAGE_BACK_CONNECT=<node4>:47712 \
#          STAGE_BACK_SWAP=1 STAGE_BACK_ROLE=both --layer-range 0:20
# ran 16 tokens at temperature 0, node4 printing " Paris, a city renowned for its rich history,
# culture, and iconic landmarks." - correct text, so the semantics below are right.
#
# The four-stage configuration in this script does NOT yet run: the head exits 1 after ~98 s with
# NSPOSIXErrorDomain Code=22 "Invalid argument". What is unproven is the middle-stage back-ring
# hop and the forward port assignment, not the roles. Do not quote a number from this script until
# it prints one.
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
  TINYTITAN_STAGE_CONNECT=node2:47742 \
  TINYTITAN_STAGE_BACK_LISTEN=47730 \
  TINYTITAN_STAGE_BACK_CONNECT=node2:47720 \
  TINYTITAN_STAGE_BACK_SWAP=1 \
  TINYTITAN_STAGE_BACK_ROLE=both \
  nohup ~/tt-bins/TinyTitanCLI --model ~/Downloads/qwen36-4bit.gturbo --prompt 'The capital of France is' \
  --max-new $TOKENS --temperature 0 --layer-range 10:20 \
  --expert-cache-slots $CACHE > /tmp/c1.out 2> /tmp/c1.err" &
sleep 3

# --- node3: first stage, embedder; drives the run ---
START=$(python3 -c 'import time;print(time.time())')
ssh node3@node3 "env TINYTITAN_STAGE_CONNECT=node1:47741 \
  TINYTITAN_STAGE_BACK_CONNECT=node1:47730 \
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
