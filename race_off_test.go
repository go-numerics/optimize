//go:build !race

package optimize

const raceEnabled = false

var raceSubset map[string]bool
