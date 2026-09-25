package com.microllm.train;

import com.microllm.model.Transformer;
import com.microllm.optim.AdamW;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Causal next-token trainer with shuffled windows, optional validation, and early stopping.
 */
public final class Trainer {
    private final Transformer model;
    private final TokenDataset trainData;
    private final TokenDataset validationData;
    private final AdamW optimizer;
    private final int patience;
    private final double baseLearningRate;
    private final int windowsPerEpoch;
    private final List<TokenDataset.AnswerAnchor> answerWindowStarts;
    private final List<TokenDataset.AnswerAnchor> validationAnswerWindowStarts;
    private final Random random;

    public Trainer(Transformer model, TokenDataset data, double learningRate) {
        this(model, data.split(0.1), learningRate, 50, 42L, 0);
    }

    public Trainer(Transformer model, TokenDataset.Split split, double learningRate, int patience, long seed) {
        this(model, split, learningRate, patience, seed, 0, List.of());
    }

    /** A zero window limit trains every disjoint window in each epoch. */
    public Trainer(Transformer model, TokenDataset.Split split, double learningRate,
                   int patience, long seed, int windowsPerEpoch) {
        this(model, split, learningRate, patience, seed, windowsPerEpoch, List.of());
    }

    /** Answer starts can be oversampled so a short prompt and its answer share the context window. */
    public Trainer(Transformer model, TokenDataset.Split split, double learningRate,
                   int patience, long seed, int windowsPerEpoch,
                   List<TokenDataset.AnswerAnchor> answerWindowStarts) {
        if (windowsPerEpoch < 0) throw new IllegalArgumentException("windowsPerEpoch cannot be negative");
        this.model = model;
        this.trainData = split.train();
        this.validationData = split.validation();
        this.optimizer = new AdamW(model.trainableParameters(), learningRate);
        this.baseLearningRate = learningRate;
        this.patience = Math.max(1, patience);
        this.windowsPerEpoch = windowsPerEpoch;
        this.answerWindowStarts = answerWindowStarts.stream()
                .filter(anchor -> anchor.windowStart() >= 0
                        && anchor.firstTargetIndex() >= 0
                        && anchor.targetLength() > anchor.firstTargetIndex()
                        && anchor.windowStart() + anchor.targetLength() < trainData.size())
                .collect(java.util.stream.Collectors.toCollection(java.util.ArrayList::new));
        int trainSize = trainData.size();
        int maxSeqLen = model.config().maxSeqLen();
        this.validationAnswerWindowStarts = validationData == null ? List.of() : answerWindowStarts.stream()
                .filter(anchor -> {
                    int answerTarget = anchor.windowStart() + anchor.firstTargetIndex() + 1;
                    return answerTarget >= trainSize && answerTarget < trainSize + validationData.size();
                })
                .map(anchor -> {
                    int absoluteTarget = anchor.windowStart() + anchor.firstTargetIndex() + 1;
                    int absoluteEnd = anchor.windowStart() + anchor.targetLength() + 1;
                    int relativeTarget = absoluteTarget - trainSize;
                    int originalStart = Math.max(0, anchor.windowStart() - trainSize);
                    int start = Math.max(originalStart, relativeTarget - (maxSeqLen - 1));
                    int relativeEnd = Math.min(validationData.size(), absoluteEnd - trainSize);
                    int targetLength = relativeEnd - start - 1;
                    return new TokenDataset.AnswerAnchor(
                            start, relativeTarget - start - 1, targetLength);
                })
                .filter(anchor -> anchor.targetLength() > anchor.firstTargetIndex())
                .toList();
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
        Collections.shuffle(answerWindowStarts, random);
        if (validationData != null) {
            System.out.printf("initial val_loss %.4f%n", evaluate(validationData, maxSeqLen));
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double progress = epochs <= 1 ? 0.0 : (double) (epoch - 1) / (epochs - 1);
            optimizer.setLearningRate(baseLearningRate * 0.5 * (1.0 + Math.cos(Math.PI * progress)));
            Collections.shuffle(offsets, random);
            int availableWindows = answerWindowStarts.isEmpty() ? offsets.size() : answerWindowStarts.size();
            int windowsThisEpoch = windowsPerEpoch == 0
                    ? availableWindows
                    : Math.min(windowsPerEpoch, availableWindows);
            int minimumCoverageEpochs = windowsPerEpoch == 0 || answerWindowStarts.isEmpty()
                    ? 1
                    : (availableWindows + windowsPerEpoch - 1) / windowsPerEpoch;
            List<WindowSelection> epochOffsets = selectEpochOffsets(offsets, windowsThisEpoch, epoch);
            double trainSum = 0.0;
            for (WindowSelection selection : epochOffsets) {
                int offset = selection.offset();
                optimizer.zeroGrad();
                TokenDataset.Window window = trainData.window(offset, selection.targetLength() < 0
                        ? maxSeqLen : selection.targetLength());
                int firstTarget = Math.min(selection.firstTargetIndex(), window.target().length - 1);
                trainSum += model.trainWindow(window.input(), window.target(), offset, firstTarget);
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
                    || epoch == epochs
                    || epochsWithoutImprovement >= patience;
            if (shouldLog) {
                if (validationData != null) {
                    System.out.printf(
                            "epoch %d/%d train_loss %.4f val_loss %.4f%n",
                            epoch,
                            epochs,
                            lastTrainLoss,
                            validationLoss);
                } else {
                    System.out.printf("epoch %d/%d loss %.4f%n", epoch, epochs, lastTrainLoss);
                }
            }

            if (epoch >= minimumCoverageEpochs && epochsWithoutImprovement >= patience) {
                System.out.printf("early stop at epoch %d (patience %d)%n", epoch, patience);
                break;
            }
        }
        if (bestParameters != null) restoreParameters(bestParameters);
        return bestMetric;
    }

    private List<WindowSelection> selectEpochOffsets(List<Integer> allOffsets, int count, int epoch) {
        List<WindowSelection> selected = new java.util.ArrayList<>(count);
        if (answerWindowStarts.isEmpty()) {
            for (int offset : allOffsets.subList(0, Math.min(count, allOffsets.size()))) selected.add(new WindowSelection(offset, 0, -1));
            return selected;
        }
        int size = answerWindowStarts.size();
        int start = windowsPerEpoch == 0 ? 0 : (int) (((long) (epoch - 1) * count) % size);
        for (int index = 0; index < Math.min(count, size); index++) {
            TokenDataset.AnswerAnchor anchor = answerWindowStarts.get((start + index) % size);
            selected.add(new WindowSelection(
                    anchor.windowStart(), anchor.firstTargetIndex(), anchor.targetLength()));
        }
        return selected;
    }

    private record WindowSelection(int offset, int firstTargetIndex, int targetLength) { }

    private double[][] snapshotParameters() {
        List<com.microllm.tensor.Tensor> parameters = model.trainableParameters();
        double[][] snapshot = new double[parameters.size()][];
        for (int i = 0; i < parameters.size(); i++) snapshot[i] = parameters.get(i).values().clone();
        return snapshot;
    }

    private void restoreParameters(double[][] snapshot) {
        List<com.microllm.tensor.Tensor> parameters = model.trainableParameters();
        for (int i = 0; i < parameters.size(); i++) {
            System.arraycopy(snapshot[i], 0, parameters.get(i).values(), 0, snapshot[i].length);
        }
    }

    private double evaluate(TokenDataset data, int maxSeqLen) {
        if (data == null) {
            return Double.NaN;
        }
        double sum = 0.0;
        int windows = 0;
        if (data == validationData && !validationAnswerWindowStarts.isEmpty()) {
            for (TokenDataset.AnswerAnchor anchor : validationAnswerWindowStarts) {
                TokenDataset.Window window = data.window(anchor.windowStart(), anchor.targetLength());
                int firstTarget = Math.min(anchor.firstTargetIndex(), window.target().length - 1);
                sum += model.evaluateWindow(window.input(), window.target(), anchor.windowStart(), firstTarget);
                windows++;
            }
        } else {
            List<Integer> offsets = data.windowOffsets(maxSeqLen);
            int sampleCount = Math.min(128, offsets.size());
            for (int sample = 0; sample < sampleCount; sample++) {
                int index = sampleCount == offsets.size() ? sample : (int) ((long) sample * offsets.size() / sampleCount);
                int offset = offsets.get(index);
                TokenDataset.Window window = data.window(offset, maxSeqLen);
                sum += model.evaluateWindow(window.input(), window.target(), offset);
                windows++;
            }
        }
        return sum / Math.max(1, windows);
    }
}
