package com.microllm.optim;
import com.microllm.tensor.Tensor;
import java.util.*;
public final class AdamW implements Optimizer {
 private final List<Tensor> parameters; private final List<double[]> first = new ArrayList<>(), second = new ArrayList<>(); private final double lr,beta1,beta2,epsilon,decay; private int step;
 public AdamW(List<Tensor> parameters, double lr) { this(parameters,lr,.9,.999,1e-8,0); }
 public AdamW(List<Tensor> parameters,double lr,double beta1,double beta2,double epsilon,double decay) { if(lr<=0) throw new IllegalArgumentException("learning rate must be positive"); this.parameters=List.copyOf(parameters);this.lr=lr;this.beta1=beta1;this.beta2=beta2;this.epsilon=epsilon;this.decay=decay; for(Tensor p:parameters){first.add(new double[p.size()]);second.add(new double[p.size()]);} }
 public void step(){ step++; for(int p=0;p<parameters.size();p++){Tensor parameter=parameters.get(p);double[] m=first.get(p),v=second.get(p),g=parameter.gradient();for(int i=0;i<parameter.size();i++){m[i]=beta1*m[i]+(1-beta1)*g[i];v[i]=beta2*v[i]+(1-beta2)*g[i]*g[i];double mh=m[i]/(1-Math.pow(beta1,step)),vh=v[i]/(1-Math.pow(beta2,step));parameter.set(i,parameter.get(i)-lr*(mh/(Math.sqrt(vh)+epsilon)+decay*parameter.get(i)));}} }
 public void zeroGrad(){for(Tensor p:parameters)p.zeroGradient();}
}
