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

# What a process costs before it does any of your work.
#
# `bench` above times 25 programs and none of them measure STARTUP, so the cost
# every `check`, `fmt`, `ast` and every edit-run cycle pays was invisible until
# someone looked: ~15ms to build the self-hosted front end, against a 2ms
# process floor. Reports rather than asserts, and keeps no baseline file, for
# the same reason jit_coverage.sh keeps none.
#
# N=40 make startup-cost   for a steadier mean; the default is 20.
.PHONY: startup-cost
startup-cost: $(TARGET)
	@./scripts/dev/startup_cost.sh
