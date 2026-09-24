package com.microllm.train;
import java.io.IOException; import java.nio.file.*; import java.util.regex.*;
/** Flat next-token dataset written by Go's tokenizer, with causal context windows. */
public final class TokenDataset {
 private final int[] tokens; private TokenDataset(int[] tokens){if(tokens.length<2)throw new IllegalArgumentException("need at least two tokens");this.tokens=tokens;}
 public static TokenDataset load(Path path)throws IOException{Matcher m=Pattern.compile("-?\\d+").matcher(Files.readString(path));java.util.ArrayList<Integer>v=new java.util.ArrayList<>();while(m.find())v.add(Integer.parseInt(m.group()));int[]t=new int[v.size()];for(int i=0;i<t.length;i++)t[i]=v.get(i);return new TokenDataset(t);}
 public int size(){return tokens.length;} public int inputAt(int i){return tokens[i];} public int targetAt(int i){return tokens[i+1];}
 /** Returns a bounded causal training window beginning at token offset. */
 public Window window(int offset,int maxLength){if(offset<0||offset>=tokens.length-1)throw new IllegalArgumentException("invalid window offset");int length=Math.min(maxLength,tokens.length-offset-1);int[]in=new int[length],out=new int[length];System.arraycopy(tokens,offset,in,0,length);System.arraycopy(tokens,offset+1,out,0,length);return new Window(in,out);}
 public record Window(int[] input,int[] target){}
}
