package com.microllm.train;
import com.microllm.model.*;
import com.microllm.export.WeightExporter;
import java.nio.file.*;
import java.io.*;
import java.util.regex.*;
public final class TrainMain {
    private static final Pattern P=Pattern.compile("^\\s*\\\"(?:\\\\.|[^\\\"\\\\])*\\\"[,]?\\s*$");
    public static void main(String[]a)throws IOException{Path v=Path.of(arg(a,"--vocab","../data/tokenized/vocab.json")),t=Path.of(arg(a,"--tokens","../data/tokenized/tokens.json")),o=Path.of(arg(a,"--output","../models/exported/model.json"));
    int n=0;for(String l:Files.readAllLines(v))if(P.matcher(l).matches())n++;
    Transformer m=new Transformer(ModelConfig.small(n),42);
    double loss=new Trainer(m,TokenDataset.load(t),Double.parseDouble(arg(a,"--learning-rate","0.03"))).train(Integer.parseInt(arg(a,"--epochs","1000")));
    WeightExporter.writeModel(o,m);System.out.printf("Wrote trained Go-compatible model: %s (loss %.4f)%n",o,loss);
    }
    private static String arg(String[]a,String f,String d){
        for(int i=0;i+1<a.length;i++)
        if(a[i].equals(f))
        return a[i+1];
        return d;
        }
        }
