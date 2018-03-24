package net.mydreamy.mlpharmaceutics.transfer;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossL1;
import org.nd4j.linalg.lossfunctions.impl.LossL2;

public class WeightedL1Loss extends LossL1 implements org.nd4j.linalg.lossfunctions.ILossFunction {
	
	double delta = 0.2;
	
	public WeightedL1Loss() {
        this(null);
    }
	
	public WeightedL1Loss(INDArray weights) {		
		super(weights);
	}
	

	@Override 
	public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		
		//invoke super to compute ScoreArray
		INDArray scoreArr = super.scoreArray(labels, preOutput, activationFn, mask);
		
		//weight scoreArray
//		scoreArr.muli(2);
		
//		System.out.println("labels" + labels);
//        System.out.println("score:" + scoreArr);
		
		//weight 
        return scoreArr;
    }
	

	
    
	//mean datapoints and labels //net.score() 
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    	 	
    	//invoke super compute score
	    double score = super.computeScore(labels, preOutput, activationFn, mask, average) / labels.columns();
	    
        //weight 
//        score = score*outputweight;
        
        return score;
    }

    @Override //indepented with score compute
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    	
//    	INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
    	
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1)
                            + ") does not match output layer" + " number of outputs (nOut = " + preOutput.size(1)
                            + ") ");
            
        }
        
        int r = labels.rows();
        int c = labels.columns();
        int masknum = 0;
        int labelp = 0;
        
        // compute weight
        for (int i = 0; i < r; i++) {
        	
    			for (int j = 0; j < c; j++) {
    				
    				if (mask.getDouble(i, j) == 1) {
    					masknum++;
    					if (labels.getDouble(i, j) == 1)
    						labelp++;
        			}
    				
    			}
        }
        
        double tp = labelp / (double ) masknum;
        double weight = 1;
        if (tp != 0) 
        		weight = 1 / tp;
        
//        System.out.println("rate: " + weight);
            
        
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        
        INDArray outSubLabels = output.sub(labels);
        
        INDArray dLda = Nd4j.getExecutioner().execAndReturn(new Sign(outSubLabels));
 //      INDArray dLda = outSubLabels.muli(2);
        
        double weight1 = 0.9;
        double weight2 = 0.1;
        
        if (weight > 1) {
     
	        //apply weight to cost
	        for (int i = 0; i < r; i++) {
	        		for (int j = 0; j < c; j++) {
	        			
	        			if (mask.getDouble(i, j) == 1) {
	        				
	        				double loss = dLda.getDouble(i, j);
	        				
	        				if ((labels.getDouble(i, j) == 1 && output.getDouble(i, j) <= 0.5)) {	        				
	        					
	        						loss = loss * weight * weight1;	        					
	        					
	        				} else {	    
	        					
	        					if (labels.getDouble(i, j) == 0 && output.getDouble(i, j) > 0.5)	{
	        							
		        					loss = loss * weight * weight2;
		        					
	        					}	
	
	        				}
	        			
		        			if (loss > 1000) {
	    						System.out.println("loss > 1: " + loss);
	    						dLda.put(i, j, 1000);
	    					} else if (loss < -1000) {
	    						System.out.println("loss < -1: " + loss);
	    						dLda.put(i, j, -1000);
	    					} else {	    						
	    					    dLda.put(i, j, loss);
	    					}
	        				
	        			}
	        			
	        		}
	        }
        
        } else {
        		
//        		System.out.println("weight < 1" + weight);
        }
              
//        System.out.println("dlda: " + dLda.shapeInfoToString());


        if (weights != null) {
            dLda.muliRowVector(weights);
        }

        if(mask != null && LossUtil.isPerOutputMasking(dLda, mask)){
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - but buy us a tiny bit of performance
            LossUtil.applyMask(dLda, mask);
        }


        //Loss function with masking
        if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
            LossUtil.applyMask(dLda, mask);
        }
        
        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO handle activation function parameter gradients
    	
//	    	System.out.println("gradients: " + gradients.shapeInfoToString());
	    	
	//    	System.out.println("compute graident" + gradients);
	    	
        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }
        
	    	//weight gradient
//	    	gradients.muli(outputweight); 
	    	
	    return gradients;
    }

}
