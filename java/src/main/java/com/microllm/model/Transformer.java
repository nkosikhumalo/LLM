package com.microllm.model;

import com.microllm.layers.FeedForward;
import com.microllm.layers.MultiHeadAttention;
import com.microllm.tensor.Ops;
import com.microllm.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Causal Transformer language model with trained attention and GELU FFN residuals.
 */
public final class Transformer {
    private final ModelConfig config;
    private final Tensor tokenEmbedding;
    private final Tensor positionEmbedding;
    private final Tensor finalNorm;
    private final Tensor output;
    /** Per layer: attn_norm, q, k, v, o, ffn_norm, ffn_in, ffn_out. */
    private final Tensor[] layerWeights;
    private final Random dropoutRandom;
    private static final double DROPOUT_RATE = 0.1;
    private boolean fakeQuantization;

    public Transformer(ModelConfig config, long seed) {
        this.config = config;
        Random random = new Random(seed);
        dropoutRandom = new Random(seed ^ 0x5DEECE66DL);
        tokenEmbedding = Tensor.randomNormal(random, 0.02, config.vocabSize(), config.dModel());
        positionEmbedding = Tensor.randomNormal(random, 0.02, config.maxSeqLen(), config.dModel());
        finalNorm = Tensor.ones(config.dModel());
        output = Tensor.randomNormal(random, 0.02, config.dModel(), config.vocabSize());
        layerWeights = new Tensor[config.nLayers() * TransformerBlock.WEIGHT_COUNT];
        for (int layer = 0; layer < config.nLayers(); layer++) {
            int base = TransformerBlock.weightOffset(layer);
            layerWeights[base] = Tensor.ones(config.dModel());
            for (int i = 1; i <= 4; i++) {
                layerWeights[base + i] = Tensor.randomNormal(random, 0.02, config.dModel(), config.dModel());
            }
            layerWeights[base + 5] = Tensor.ones(config.dModel());
            layerWeights[base + 6] = Tensor.randomNormal(random, 0.02, config.dModel(), config.dFfn());
            layerWeights[base + 7] = Tensor.randomNormal(random, 0.02, config.dFfn(), config.dModel());
        }
    }

    public ModelConfig config() {
        return config;
    }

    public Tensor tokenEmbedding() {
        return tokenEmbedding;
    }

    public Tensor positionEmbedding() {
        return positionEmbedding;
    }

    public Tensor finalNorm() {
        return finalNorm;
    }

    public Tensor output() {
        return output;
    }

    public Tensor layerWeight(int index) {
        return layerWeights[index];
    }

    public void setFakeQuantization(boolean enabled) {
        fakeQuantization = enabled;
    }

    public void loadParameters(List<double[]> parameters) {
        List<Tensor> expected = trainableParameters();
        if (parameters.size() != expected.size()) throw new IllegalArgumentException("checkpoint tensor count mismatch");
        for (int i = 0; i < expected.size(); i++) {
            if (parameters.get(i).length != expected.get(i).size()) {
                throw new IllegalArgumentException("checkpoint tensor " + i + " has an invalid size");
            }
            System.arraycopy(parameters.get(i), 0, expected.get(i).values(), 0, expected.get(i).size());
        }
    }

    /** Symmetric signed INT8 fake quantization with a straight-through gradient. */
    public static double[] fakeQuantizeWeights(double[] values) {
        return fakeQuantizeSegment(values, 0, values.length);
    }

    /** Fake-quantizes each first-dimension channel using the PTQ per-channel rule. */
    public static double[] fakeQuantizeWeights(double[] values, int... shape) {
        if (shape.length < 2) return fakeQuantizeWeights(values);
        int channels = shape[0];
        int stride = 1;
        for (int axis = 1; axis < shape.length; axis++) stride = Math.multiplyExact(stride, shape[axis]);
        if (channels < 1 || channels * stride != values.length) throw new IllegalArgumentException("weight shape does not match values");
        double[] result = new double[values.length];
        for (int channel = 0; channel < channels; channel++) {
            double[] slice = java.util.Arrays.copyOfRange(values, channel * stride, (channel + 1) * stride);
            double[] quantized = fakeQuantizeSegment(slice, 0, slice.length);
            System.arraycopy(quantized, 0, result, channel * stride, stride);
        }
        return result;
    }

    private static double[] fakeQuantizeSegment(double[] values, int start, int length) {
        double maxAbs = 0.0;
        for (int i = start; i < start + length; i++) {
            double value = values[i];
            if (!Double.isFinite(value)) throw new IllegalArgumentException("weight contains a non-finite value");
            maxAbs = Math.max(maxAbs, Math.abs(value));
        }
        double scale = maxAbs == 0.0 ? 1.0 : maxAbs / 127.0;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            double quantized = Math.max(-127, Math.min(127, Math.rint(values[start + i] / scale)));
            result[i] = quantized * scale;
        }
        return result;
    }

    private double[] forwardWeights(Tensor tensor) {
        return fakeQuantization ? fakeQuantizeWeights(tensor.values(), tensor.shape()) : tensor.values();
    }

    public List<Tensor> trainableParameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.add(tokenEmbedding);
        parameters.add(positionEmbedding);
        parameters.add(finalNorm);
        parameters.add(output);
        for (int layer = 0; layer < config.nLayers(); layer++) {
            int base = TransformerBlock.weightOffset(layer);
            parameters.add(layerWeights[base]);     // attn_norm
            parameters.add(layerWeights[base + 1]); // q
            parameters.add(layerWeights[base + 2]); // k
            parameters.add(layerWeights[base + 3]); // v
            parameters.add(layerWeights[base + 4]); // o
            parameters.add(layerWeights[base + 5]); // ffn_norm
            parameters.add(layerWeights[base + 6]); // ffn_in
            parameters.add(layerWeights[base + 7]); // ffn_out
        }
        return parameters;
    }

    public double trainWindow(int[] input, int[] target, int start) {
        return runWindow(input, target, start, true);
    }

    public double evaluateWindow(int[] input, int[] target, int start) {
        return runWindow(input, target, start, false);
    }

    private double runWindow(int[] input, int[] target, int start, boolean train) {
        if (input.length == 0 || input.length != target.length || input.length > config.maxSeqLen()) {
            throw new IllegalArgumentException("invalid window");
        }
        if (start < 0) throw new IllegalArgumentException("window start must be nonnegative");
        for (int token : input) {
            if (token < 0 || token >= config.vocabSize()) throw new IllegalArgumentException("input token " + token + " outside vocabulary size " + config.vocabSize());
        }
        for (int token : target) {
            if (token < 0 || token >= config.vocabSize()) throw new IllegalArgumentException("target token outside vocabulary");
        }
        int n = input.length;
        int supervisedTokens = n;
        int d = config.dModel();
        int v = config.vocabSize();
        double[] tokenWeights = forwardWeights(tokenEmbedding);
        double[] positionWeights = forwardWeights(positionEmbedding);
        double[] finalNormWeights = forwardWeights(finalNorm);
        double[] outputWeights = forwardWeights(output);
        double[][] layerForwardWeights = new double[layerWeights.length][];
        for (int i = 0; i < layerWeights.length; i++) layerForwardWeights[i] = forwardWeights(layerWeights[i]);

        double[][] x = new double[n][d];
        for (int p = 0; p < n; p++) {
            int position = p;
            for (int i = 0; i < d; i++) {
                x[p][i] = tokenWeights[input[p] * d + i] + positionWeights[position * d + i];
            }
        }

        LayerCache[] layers = new LayerCache[config.nLayers()];
        for (int layer = 0; layer < config.nLayers(); layer++) {
            int base = TransformerBlock.weightOffset(layer);
            double[][] residualIn = copyRows(x);
            double[][] attnNormed = new double[n][d];
            for (int p = 0; p < n; p++) {
                attnNormed[p] = rms(x[p], layerForwardWeights[base]);
            }
            MultiHeadAttention attention = new MultiHeadAttention(
                    d,
                    config.nHeads(),
                    layerForwardWeights[base + 1],
                    layerForwardWeights[base + 2],
                    layerForwardWeights[base + 3],
                    layerForwardWeights[base + 4]);
            MultiHeadAttention.Cache attnCache = attention.forward(attnNormed, train ? DROPOUT_RATE : 0.0, dropoutRandom);
            double[][] afterAttn = new double[n][d];
            for (int p = 0; p < n; p++) {
                for (int i = 0; i < d; i++) {
                    afterAttn[p][i] = residualIn[p][i] + attnCache.output()[p][i];
                }
            }
            double[][] ffnNormed = new double[n][d];
            for (int p = 0; p < n; p++) {
                ffnNormed[p] = rms(afterAttn[p], layerForwardWeights[base + 5]);
            }
            FeedForward ffn = new FeedForward(
                    d,
                    config.dFfn(),
                    layerForwardWeights[base + 6],
                    layerForwardWeights[base + 7]);
            FeedForward.Cache ffnCache = ffn.forward(ffnNormed, train ? DROPOUT_RATE : 0.0, dropoutRandom);
            double[][] next = new double[n][d];
            for (int p = 0; p < n; p++) {
                for (int i = 0; i < d; i++) {
                    next[p][i] = afterAttn[p][i] + ffnCache.output()[p][i];
                }
            }
            layers[layer] = new LayerCache(residualIn, attnCache, afterAttn, ffnCache);
            x = next;
        }

        double loss = 0.0;
        double[][] gradX = new double[n][d];
        for (int p = 0; p < n; p++) {
            double[] normalized = rms(x[p], finalNormWeights);
            double[] probabilities = Ops.softmax(row(normalized, outputWeights, d, v));
            loss -= Math.log(Math.max(probabilities[target[p]], 1e-12));
            if (train) {
                probabilities[target[p]] -= 1.0;
                double[] gradNorm = new double[d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < v; j++) {
                        double gradient = probabilities[j] / supervisedTokens;
                        output.gradient()[i * v + j] += normalized[i] * gradient;
                        gradNorm[i] += outputWeights[i * v + j] * gradient;
                    }
                }
                gradX[p] = rmsBack(gradNorm, x[p], finalNorm, finalNormWeights);
            }
        }
        if (!train) {
            return loss / supervisedTokens;
        }

        for (int layer = config.nLayers() - 1; layer >= 0; layer--) {
            int base = TransformerBlock.weightOffset(layer);
            LayerCache cache = layers[layer];

            FeedForward ffn = new FeedForward(
                    d,
                    config.dFfn(),
                    layerForwardWeights[base + 6],
                    layerForwardWeights[base + 7]);
            FeedForward.Gradients ffnGrad = ffn.backward(cache.ffn(), gradX);
            add(layerWeights[base + 6].gradient(), ffnGrad.inWeight());
            add(layerWeights[base + 7].gradient(), ffnGrad.outWeight());

            double[][] gradAfterAttn = new double[n][d];
            for (int p = 0; p < n; p++) {
                double[] fromFfn = rmsBack(ffnGrad.input()[p], cache.afterAttn()[p], layerWeights[base + 5], layerForwardWeights[base + 5]);
                for (int i = 0; i < d; i++) {
                    gradAfterAttn[p][i] = gradX[p][i] + fromFfn[i];
                }
            }

            MultiHeadAttention attention = new MultiHeadAttention(
                    d,
                    config.nHeads(),
                    layerForwardWeights[base + 1],
                    layerForwardWeights[base + 2],
                    layerForwardWeights[base + 3],
                    layerForwardWeights[base + 4]);
            MultiHeadAttention.Gradients attnGrad = attention.backward(cache.attention(), gradAfterAttn);
            add(layerWeights[base + 1].gradient(), attnGrad.q());
            add(layerWeights[base + 2].gradient(), attnGrad.k());
            add(layerWeights[base + 3].gradient(), attnGrad.v());
            add(layerWeights[base + 4].gradient(), attnGrad.o());

            double[][] nextGrad = new double[n][d];
            for (int p = 0; p < n; p++) {
                double[] fromAttn = rmsBack(attnGrad.input()[p], cache.residualIn()[p], layerWeights[base], layerForwardWeights[base]);
                for (int i = 0; i < d; i++) {
                    nextGrad[p][i] = gradAfterAttn[p][i] + fromAttn[i];
                }
            }
            gradX = nextGrad;
        }

        for (int p = 0; p < n; p++) {
            int position = p;
            for (int i = 0; i < d; i++) {
                tokenEmbedding.gradient()[input[p] * d + i] += gradX[p][i];
                positionEmbedding.gradient()[position * d + i] += gradX[p][i];
            }
        }
        return loss / supervisedTokens;
    }

    private double[] rms(double[] x, double[] scale) {
        double sumSquares = 0.0;
        for (double value : x) {
            sumSquares += value * value;
        }
        double inv = 1.0 / Math.sqrt(sumSquares / x.length + config.rmsNormEpsilon());
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * inv * scale[i];
        }
        return y;
    }

    private double[] rmsBack(double[] grad, double[] x, Tensor trainableScale, double[] scale) {
        double sumSquares = 0.0;
        double dot = 0.0;
        for (int i = 0; i < x.length; i++) {
            sumSquares += x[i] * x[i];
            dot += grad[i] * scale[i] * x[i];
        }
        double inv = 1.0 / Math.sqrt(sumSquares / x.length + config.rmsNormEpsilon());
        double[] dx = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            trainableScale.gradient()[i] += x[i] * inv * grad[i];
            dx[i] = inv * grad[i] * scale[i] - x[i] * dot * inv * inv * inv / x.length;
        }
        return dx;
    }

    private static double[] row(double[] x, double[] matrix, int rows, int cols) {
        double[] y = new double[cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y[j] += x[i] * matrix[i * cols + j];
            }
        }
        return y;
    }

    private static void add(double[] target, double[] source) {
        for (int i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }

    private static double[][] copyRows(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }

    private record LayerCache(
            double[][] residualIn,
            MultiHeadAttention.Cache attention,
            double[][] afterAttn,
            FeedForward.Cache ffn) {}
}
