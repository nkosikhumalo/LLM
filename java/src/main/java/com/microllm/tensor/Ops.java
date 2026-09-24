package com.microllm.tensor;
public final class Ops { private Ops(){} public static double[] softmax(double[] x){double max=Double.NEGATIVE_INFINITY;for(double v:x)max=Math.max(max,v);double sum=0;double[]p=new double[x.length];for(int i=0;i<x.length;i++){p[i]=Math.exp(x[i]-max);sum+=p[i];}for(int i=0;i<p.length;i++)p[i]/=sum;return p;} }
