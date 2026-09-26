package com.microllm.train;

import com.microllm.model.Transformer;
import com.microllm.optim.AdamW;
import com.microllm.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/** Causal next-token trainer with shuffled windows, validation, and early stopping. */
public final class Trainer {
    private final Transformer model;
    private final TokenDataset trainData;
    private final TokenDataset validationData;
    private final AdamW optimizer;
    private final int patience;
    private final double baseLearningRate;
    private final int windowsPerEpoch;
    private final Random random;

    public Trainer(Transformer model, TokenDataset data, double learningRate) {
        this(model, data.split(0.1), learningRate, 50, 42L, 0);
    }

    public Trainer(Transformer model, TokenDataset.Split split, double learningRate, int patience, long seed) {
        this(model, split, learningRate, patience, seed, 0);
    }

    /** A zero window limit trains every causal window in each epoch. */
    public Trainer(Transformer model, TokenDataset.Split split, double learningRate,
                   int patience, long seed, int windowsPerEpoch) {
        if (windowsPerEpoch < 0) throw new IllegalArgumentException("windowsPerEpoch cannot be negative");
        this.model = model;
        this.trainData = split.train();
        this.validationData = split.validation();
        this.optimizer = new AdamW(model.trainableParameters(), learningRate);
        this.baseLearningRate = learningRate;
        this.patience = Math.max(1, patience);
        this.windowsPerEpoch = windowsPerEpoch;
        this.random = new Random(seed);
    }

    public double train(int epochs) {
        if (epochs < 1) throw new IllegalArgumentException("epochs must be positive");
        double bestMetric = Double.POSITIVE_INFINITY;
        double lastTrainLoss = 0.0;
        double[][] bestParameters = null;
        int epochsWithoutImprovement = 0;
        int maxSeqLen = model.config().maxSeqLen();
        List<Integer> offsets = trainData.windowOffsets(maxSeqLen);
        if (validationData != null) {
            System.out.printf("initial val_loss %.4f%n", evaluate(validationData, maxSeqLen));
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double progress = epochs <= 1 ? 0.0 : (double) (epoch - 1) / (epochs - 1);
            optimizer.setLearningRate(baseLearningRate * 0.5 * (1.0 + Math.cos(Math.PI * progress)));
            Collections.shuffle(offsets, random);
            int windowsThisEpoch = windowsPerEpoch == 0
                    ? offsets.size() : Math.min(windowsPerEpoch, offsets.size());
            double trainSum = 0.0;
            for (int i = 0; i < windowsThisEpoch; i++) {
                int offset = offsets.get(i);
                optimizer.zeroGrad();
                TokenDataset.Window window = trainData.window(offset, maxSeqLen);
                trainSum += model.trainWindow(window.input(), window.target(), offset);
                optimizer.step();
            }
            lastTrainLoss = trainSum / Math.max(1, windowsThisEpoch);

            double validationLoss = evaluate(validationData, maxSeqLen);
            double metric = validationData != null ? validationLoss : lastTrainLoss;
            if (metric < bestMetric - 1e-6) {
                bestMetric = metric;
                bestParameters = snapshotParameters();
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
            }

            boolean shouldLog = epoch % Math.max(1, epochs / 10) == 0
                    || epoch == epochs || epochsWithoutImprovement >= patience;
            if (shouldLog) {
                if (validationData != null) {
                    System.out.printf("epoch %d/%d train_loss %.4f val_loss %.4f%n", epoch, epochs, lastTrainLoss, validationLoss);
                } else {
                    System.out.printf("epoch %d/%d loss %.4f%n", epoch, epochs, lastTrainLoss);
                }
            }
            if (epochsWithoutImprovement >= patience) {
                System.out.printf("early stop at epoch %d (patience %d)%n", epoch, patience);
                break;
            }
        }
        if (bestParameters != null) restoreParameters(bestParameters);
        return bestMetric;
    }

    private double[][] snapshotParameters() {
        List<Tensor> parameters = model.trainableParameters();
        double[][] snapshot = new double[parameters.size()][];
        for (int i = 0; i < parameters.size(); i++) snapshot[i] = parameters.get(i).values().clone();
        return snapshot;
    }

    private void restoreParameters(double[][] snapshot) {
        List<Tensor> parameters = model.trainableParameters();
        for (int i = 0; i < parameters.size(); i++) {
            System.arraycopy(snapshot[i], 0, parameters.get(i).values(), 0, snapshot[i].length);
        }
    }

    private double evaluate(TokenDataset data, int maxSeqLen) {
        if (data == null) return Double.NaN;
        double sum = 0.0;
        List<Integer> offsets = data.windowOffsets(maxSeqLen);
        int sampleCount = Math.min(128, offsets.size());
        for (int sample = 0; sample < sampleCount; sample++) {
            int index = sampleCount == offsets.size() ? sample : (int) ((long) sample * offsets.size() / sampleCount);
            int offset = offsets.get(index);
            TokenDataset.Window window = data.window(offset, maxSeqLen);
            sum += model.evaluateWindow(window.input(), window.target(), offset);
        }
        return sum / Math.max(1, sampleCount);
    }
}
