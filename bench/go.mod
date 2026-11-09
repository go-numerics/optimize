module github.com/go-numerics/optimize/bench

go 1.27.3

require (
	github.com/go-ndarray/ndarray v0.34.0
	github.com/go-numerics/optimize v0.0.0
	github.com/go-scikit/refbench v0.1.2
	github.com/go-simd/floats v0.13.0
)

require (
	github.com/go-fleettools/fleettools v0.81.0 // indirect
	github.com/go-sparse/sparse v0.3.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

replace github.com/go-numerics/optimize => ../
