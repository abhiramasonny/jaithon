# Benchmarks, and the bare package names that select which suite runs.

#: Under the GPU lock, because several agents share this machine and any two
#: measuring at once produce numbers that mean nothing: this suite read 1.98x
#: under a load average of 4.5 and 3.10x on the same commit with the machine
#: quiet, with individual rows moving by a factor of three.
bench: $(TARGET)
	@LEVEL="$(LEVEL)" ./scripts/bench/gpu_lock.sh ./scripts/bench/run_bench.sh $(filter jaitensor jaicv jainum jaiframe,$(MAKECMDGOALS))

jaitensor:
	@:

jaicv:
	@:

jainum:
	@:

jaiframe:
	@:
