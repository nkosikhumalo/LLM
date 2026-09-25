// Package sampler selects a token from a model's next-token logits.
package sampler

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

type Config struct {
	Temperature       float64
	TopP              float64
	TopK              int
	Greedy            bool
	RepetitionPenalty float64
	SeenTokens        []int
}

func DefaultConfig() Config { return Config{Temperature: 1, TopP: 1} }

type candidate struct {
	tokenID     int
	probability float64
}

// Sample applies temperature, top-k, and nucleus filtering before sampling.
func Sample(logits []float64, config Config, randomSource *rand.Rand) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("cannot sample empty logits")
	}
	if config.Temperature < 0 || config.TopK < 0 || config.TopP < 0 || config.TopP > 1 {
		return 0, fmt.Errorf("invalid sampling configuration")
	}
	if config.RepetitionPenalty != 0 && config.RepetitionPenalty < 1 {
		return 0, fmt.Errorf("repetition penalty must be at least 1")
	}
	adjusted := append([]float64(nil), logits...)
	if config.RepetitionPenalty > 1 {
		seen := make(map[int]struct{}, len(config.SeenTokens))
		for _, tokenID := range config.SeenTokens {
			if tokenID < 0 || tokenID >= len(adjusted) {
				continue
			}
			if _, alreadySeen := seen[tokenID]; alreadySeen {
				continue
			}
			seen[tokenID] = struct{}{}
			if adjusted[tokenID] > 0 {
				adjusted[tokenID] /= config.RepetitionPenalty
			} else {
				adjusted[tokenID] *= config.RepetitionPenalty
			}
		}
	}
	mostLikelyID := 0
	for tokenID := 1; tokenID < len(adjusted); tokenID++ {
		if adjusted[tokenID] > adjusted[mostLikelyID] {
			mostLikelyID = tokenID
		}
	}
	if config.Greedy || config.Temperature == 0 {
		return mostLikelyID, nil
	}

	maximumLogit := math.Inf(-1)
	for _, logit := range adjusted {
		if scaledLogit := logit / config.Temperature; scaledLogit > maximumLogit {
			maximumLogit = scaledLogit
		}
	}
	candidates := make([]candidate, len(logits))
	probabilitySum := 0.0
	for tokenID, logit := range adjusted {
		probability := math.Exp(logit/config.Temperature - maximumLogit)
		candidates[tokenID] = candidate{tokenID, probability}
		probabilitySum += probability
	}
	for index := range candidates {
		candidates[index].probability /= probabilitySum
	}
	sort.Slice(candidates, func(left, right int) bool { return candidates[left].probability > candidates[right].probability })

	candidateCount := len(candidates)
	if config.TopK > 0 && config.TopK < candidateCount {
		candidateCount = config.TopK
	}
	if config.TopP > 0 && config.TopP < 1 {
		cumulativeProbability := 0.0
		for index := 0; index < candidateCount; index++ {
			cumulativeProbability += candidates[index].probability
			if cumulativeProbability >= config.TopP {
				candidateCount = index + 1
				break
			}
		}
	}
	probabilitySum = 0
	for index := 0; index < candidateCount; index++ {
		probabilitySum += candidates[index].probability
	}
	if randomSource == nil {
		randomSource = rand.New(rand.NewSource(1))
	}
	target, cumulativeProbability := randomSource.Float64(), 0.0
	for index := 0; index < candidateCount; index++ {
		cumulativeProbability += candidates[index].probability / probabilitySum
		if target <= cumulativeProbability {
			return candidates[index].tokenID, nil
		}
	}
	return candidates[candidateCount-1].tokenID, nil
}
