package com.microllm.model;

/** Immutable hyperparameters shared by the Java exporter and Go inference engine. */
public record ModelConfig(
        int vocabSize,
        int dModel,
        int nLayers,
        int nHeads,
        int dFfn,
        int maxSeqLen,
        double rmsNormEpsilon) {

    public ModelConfig {
        if (vocabSize < 1 || dModel < 1 || nLayers < 1 || nHeads < 1 || dFfn < 1 || maxSeqLen < 1) {
            throw new IllegalArgumentException("all model dimensions must be positive");
        }
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must divide evenly across nHeads");
        }
        if (rmsNormEpsilon <= 0.0 || !Double.isFinite(rmsNormEpsilon)) {
            throw new IllegalArgumentException("rmsNormEpsilon must be finite and positive");
        }
    }

    public static ModelConfig small(int vocabSize) {
        return new ModelConfig(vocabSize, 32, 2, 4, 64, 128, 1e-5);
    }
}
