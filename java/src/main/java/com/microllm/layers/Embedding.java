package com.microllm.layers;

/**
 * Belongs here: embedding layer — discrete token IDs → continuous vectors (d_model).
 *
 * Intended contents:
 * - Weight table [vocabSize, dModel] (~256 rows for character-level)
 * - Forward: lookup / gather rows for a sequence of IDs
 * - Backward: scatter gradients into the embedding table
 *
 * Used by: model.Transformer
 */
public class Embedding {
}
