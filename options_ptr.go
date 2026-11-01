package optimize

// Float returns a pointer to v, for the options where 0 is a meaningful value
// and the default is not 0 (DE Tol and Recombination, basin hopping T).
func Float(v float64) *float64 { return &v }

// Int returns a pointer to v, for the options where 0 is a meaningful value
// and the default is not 0 (basin hopping NIter).
func Int(v int) *int { return &v }
