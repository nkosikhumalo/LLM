package com.microllm.model;

/**
 * Belongs here: one Transformer block = attention + FFN + norms (+ residuals).
 *
 * Intended contents:
 * - MultiHeadAttention, FeedForward, LayerNorm/RMSNorm instances
 * - Forward (and backward via autograd) for a single layer
 *
 * Stacked by Transformer.
 */
public class TransformerBlock {
}
