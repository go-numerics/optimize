package optimize

import (
	"sync"
	"sync/atomic"
)

// effectiveWorkers is the number of goroutines parallelFor uses for n items.
func effectiveWorkers(n, workers int) int {
	return max(1, min(workers, n))
}

// parallelFor calls body(worker, i) for every i in [0, n). With one worker it
// is a plain loop in index order. With more, the workers take indices from an
// atomic counter; body must write its result into a slot owned by i (and may
// use scratch owned by worker, 0 <= worker < effectiveWorkers(n, workers)),
// so the outcome does not depend on scheduling.
//
// A panic in body is recovered in its worker, the other workers stop taking
// new indices, and the panic of the lowest index seen is re-raised on the
// calling goroutine with its original value: a panicking objective is
// catchable by the caller with one worker or eight.
func parallelFor(n, workers int, body func(worker, i int)) {
	w := effectiveWorkers(n, workers)
	if w == 1 {
		for i := range n {
			body(0, i)
		}
		return
	}
	var (
		next     atomic.Int64
		stop     atomic.Bool
		wg       sync.WaitGroup
		mu       sync.Mutex
		panicked bool
		panicIdx int
		panicVal any
	)
	for k := range w {
		wg.Go(func() {
			cur := 0
			defer func() {
				if v := recover(); v != nil {
					stop.Store(true)
					mu.Lock()
					if !panicked || cur < panicIdx {
						panicked, panicIdx, panicVal = true, cur, v
					}
					mu.Unlock()
				}
			}()
			for !stop.Load() {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				cur = i
				body(k, i)
			}
		})
	}
	wg.Wait()
	if panicked {
		panic(panicVal)
	}
}

// evalBatch computes out[i] = f(points[i]) for every i, with the given
// number of workers (<= 1: a plain loop). Results land in their own slots,
// so they are identical for every worker count; the objective must be safe
// for concurrent use when workers > 1. A panic in f is re-raised on the
// calling goroutine.
func evalBatch(f func([]float64) float64, points [][]float64, out []float64, workers int) {
	parallelFor(len(points), workers, func(_, i int) { out[i] = f(points[i]) })
}
