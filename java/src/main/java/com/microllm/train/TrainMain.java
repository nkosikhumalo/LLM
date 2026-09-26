package com.microllm.train;

import com.microllm.export.CheckpointImporter;
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
        double learningRate = Double.parseDouble(arg(args, "--learning-rate", "0.0003"));
        int epochs = Integer.parseInt(arg(args, "--epochs", "1000"));
        double validationFraction = Double.parseDouble(arg(args, "--val-fraction", "0.1"));
        int patience = Integer.parseInt(arg(args, "--patience", "50"));
        int windowsPerEpoch = Integer.parseInt(arg(args, "--windows-per-epoch", "0"));
        String initialWeightsArgument = arg(args, "--init-weights", "");
        boolean fakeQuantization = Boolean.parseBoolean(arg(args, "--fake-quantization", "false"));

        int vocabSize = readVocabularySize(vocabPath);
        if (vocabSize == 0) throw new IOException("vocabulary must contain at least one token");

        Transformer model;
        if (!initialWeightsArgument.isBlank()) {
            CheckpointImporter.LoadedCheckpoint initial = CheckpointImporter.read(Path.of(initialWeightsArgument));
            if (initial.config().vocabSize() != vocabSize) throw new IOException("initial checkpoint vocabulary does not match vocab.json");
            model = new Transformer(initial.config(), 42);
            model.loadParameters(initial.tensors());
        } else {
            if (fakeQuantization) throw new IOException("fake quantization requires --init-weights");
            model = new Transformer(ModelConfig.small(vocabSize), 42);
        }
        model.setFakeQuantization(fakeQuantization);
        TokenDataset dataset = TokenDataset.load(tokensPath);
        TokenDataset.Split split = dataset.split(validationFraction);
        double loss = new Trainer(model, split, learningRate, patience, 42L,
                windowsPerEpoch).train(epochs);
        WeightExporter.writeModel(outputPath, model);
        System.out.printf("Wrote %s Go-compatible model: %s (loss %.4f)%n", fakeQuantization ? "QAT fine-tuned" : "trained", outputPath, loss);
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
