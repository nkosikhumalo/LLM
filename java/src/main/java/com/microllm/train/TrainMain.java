package com.microllm.train;

import com.microllm.export.WeightExporter;
import com.microllm.model.ModelConfig;
import com.microllm.model.Transformer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class TrainMain {
    private static final Pattern VOCAB_ARRAY = Pattern.compile("\"tokens\"\\s*:\\s*\\[");

    public static void main(String[] args) throws IOException {
        Path vocabPath = Path.of(arg(args, "--vocab", "../data/tokenized/vocab.json"));
        Path tokensPath = Path.of(arg(args, "--tokens", "../data/tokenized/tokens.json"));
        Path outputPath = Path.of(arg(args, "--output", "../models/exported/model.json"));
        String anchorsArgument = arg(args, "--answer-anchors", "../data/processed/train_answer_anchors.offsets");
        Path anchorsPath = anchorsArgument.isBlank() ? null : Path.of(anchorsArgument);
        double learningRate = Double.parseDouble(arg(args, "--learning-rate", "0.0003"));
        int epochs = Integer.parseInt(arg(args, "--epochs", "1000"));
        double validationFraction = Double.parseDouble(arg(args, "--val-fraction", "0.1"));
        int patience = Integer.parseInt(arg(args, "--patience", "50"));
        int windowsPerEpoch = Integer.parseInt(arg(args, "--windows-per-epoch", "0"));

        int vocabSize = readVocabularySize(vocabPath);
        if (vocabSize == 0) throw new IOException("vocabulary must contain at least one token");

        Transformer model = new Transformer(ModelConfig.small(vocabSize), 42);
        TokenDataset dataset = TokenDataset.load(tokensPath);
        TokenDataset.Split split = dataset.split(validationFraction);
        var answerAnchors = anchorsPath != null && Files.exists(anchorsPath)
                ? TokenDataset.loadAnswerAnchors(anchorsPath)
                : java.util.List.<TokenDataset.AnswerAnchor>of();
        double loss = new Trainer(model, split, learningRate, patience, 42L,
                windowsPerEpoch, answerAnchors).train(epochs);
        WeightExporter.writeModel(outputPath, model);
        System.out.printf("Wrote trained Go-compatible model: %s (loss %.4f)%n", outputPath, loss);
    }

    private static int readVocabularySize(Path path) throws IOException {
        String json = Files.readString(path);
        Matcher array = VOCAB_ARRAY.matcher(json);
        if (!array.find()) throw new IOException("vocabulary JSON is missing its tokens array");
        int count = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int index = array.end(); index < json.length(); index++) {
            char character = json.charAt(index);
            if (inString) {
                if (escaped) escaped = false;
                else if (character == '\\') escaped = true;
                else if (character == '\"') inString = false;
            } else if (character == '\"') {
                inString = true;
                count++;
            } else if (character == ']') {
                return count;
            }
        }
        throw new IOException("vocabulary tokens array is not terminated");
    }

    private static String arg(String[] args, String flag, String defaultValue) {
        for (int i = 0; i + 1 < args.length; i++) {
            if (args[i].equals(flag)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
