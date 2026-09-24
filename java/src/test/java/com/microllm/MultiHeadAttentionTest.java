package com.microllm;
import com.microllm.layers.MultiHeadAttention;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class MultiHeadAttentionTest {
 @Test void futureTokensCannotChangeEarlierOutput(){double[] id={1,0,0,1};var attention=new MultiHeadAttention(2,1,id,id,id,id);var first=attention.forward(new double[][]{{1,0},{0,1}}).output();var changed=attention.forward(new double[][]{{1,0},{100,100}}).output();assertArrayEquals(first[0],changed[0],1e-12);assertArrayEquals(new double[]{1,0},first[0],1e-12);}
}
