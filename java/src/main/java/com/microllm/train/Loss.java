package com.microllm.train;

/**
 * Belongs here: cross-entropy (or NLL) loss for next-token prediction.
 *
 * Intended contents:
 * - Compare model logits vs target token IDs
 * - Average over batch / sequence
 * - Hook into autograd so backward starts from this scalar
 *
 * Used by: Trainer
 */
public class Loss {
}
