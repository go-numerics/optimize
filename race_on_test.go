//go:build race

package optimize

// raceEnabled is true when the tests run under the race detector (the CI
// lane that also measures coverage, where every test costs ten to twenty
// times its normal time and the whole suite is close to go test's 10 minute
// limit). The statistical tests, which have no concurrency to examine, run
// on a few cheap functions only there; the plain lanes run them all.
const raceEnabled = true

var raceSubset = map[string]bool{"Branin01/2": true, "CrossInTray/2": true, "Bohachevsky1/2": true, "Griewank/2": true}
