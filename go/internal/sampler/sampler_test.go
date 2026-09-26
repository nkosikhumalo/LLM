package sampler_test

import (
	"math/rand"
	"testing"

	"github.com/nkosikhumalo/microllm/go/internal/sampler"
)

func TestGreedyPicksArgmax(t *testing.T) {
	id, err := sampler.Sample([]float64{0.1, 3.0, 0.2}, sampler.Config{Greedy: true}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("got %d", id)
	}
}

func TestTemperatureZeroIsGreedy(t *testing.T) {
	id, err := sampler.Sample([]float64{-1, 5, 0}, sampler.Config{Temperature: 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("got %d", id)
	}
}

func TestTopKRestrictsChoices(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	counts := map[int]int{}
	for i := 0; i < 200; i++ {
		id, err := sampler.Sample([]float64{10, 9, -100}, sampler.Config{Temperature: 1, TopK: 2}, rng)
		if err != nil {
			t.Fatal(err)
		}
		counts[id]++
	}
	if counts[2] != 0 {
		t.Fatalf("top-k allowed token 2: %#v", counts)
	}
	if counts[0] == 0 || counts[1] == 0 {
		t.Fatalf("expected both top tokens, got %#v", counts)
	}
}
