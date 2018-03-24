package net.mydreamy.mlpharmaceutics.transfer;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossL1;
import org.nd4j.linalg.lossfunctions.impl.LossL2;

public class WeightedL2Loss extends LossL2 implements org.nd4j.linalg.lossfunctions.ILossFunction {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double outputweight;
	
	
	public WeightedL2Loss() {
        this(null);
    }
	
	public WeightedL2Loss(INDArray weights) {		
		super(weights);
		this.outputweight = 1;
	}
	
	public WeightedL2Loss(INDArray weights, double outputweight) {
	
		super(weights);
		
		this.outputweight = outputweight;
		
	}
	
	public WeightedL2Loss(double outputweight) {
		
		this.outputweight = outputweight;
		
	}

	@Override 
	public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		
		//invoke super to compute ScoreArray
		INDArray scoreArr = super.scoreArray(labels, preOutput, activationFn, mask);
		
		//weight scoreArray
		scoreArr.muli(2);
		
//		System.out.println("labels" + labels);
//        System.out.println("score:" + scoreArr);
		
		//weight 
        return scoreArr;
    }
	

	
//	//mean all labels
//    @Override
//    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
//    	
//      	//invoke super computeScoreArray
//        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
//        
//        //weight scoreArray
//        scoreArr.muli(outputweight);
//        
//        INDArray meanalllabels = scoreArr.sum(1);
//        
//        System.out.println("mean score: " + meanalllabels);
//        
//        return meanalllabels;
//    }
    
	//mean datapoints and labels //net.score() 
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    	 	
    	//invoke super compute score
	    double score = super.computeScore(labels, preOutput, activationFn, mask, average) / labels.columns();
	    
//	    System.out.println("mean all labels and data points: " + score);

        //weight 
        score = score*outputweight;
        
        return score;
    }

    @Override //indepented with score compute
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    	
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
        double weight = 1 / tp;
        
//        System.out.println("rate: " + weight);
        
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray dLda = output.subi(labels).muli(2);
        
        if (weight > 1) {
        
	        //apply weight to cost
	        for (int i = 0; i < r; i++) {
	        		for (int j = 0; j < c; j++) {
	        			
	        			if (mask.getDouble(i, j) == 1) {
	        				
	//        				System.out.println("output: " + output.getDouble(i, j));
	        				
	        				
	//        				System.out.println("before: " + dLda.getDouble(i, j));
	        				if (labels.getDouble(i, j) == 1) {
	        					dLda.put(i, j, dLda.getDouble(i, j) * 100);
	        				}
	        				else {
	//        					dLda.put(i, j, dLda.getDouble(i, j) * 71.42);
	        				
	        					if (labels.getDouble(i, j) == 0 && output.getDouble(i, j) > 0.5)
	        						dLda.put(i, j, dLda.getDouble(i, j) * 1);
	        				
	//        				System.out.println("after: " + dLda.getDouble(i, j));
	        				}
	        			}
	        			
	        		}
	        }
        
        }
              
//        System.out.println("dlda: " + dLda.shapeInfoToString());


        if (weights != null) {
            dLda.muliRowVector(weights);
        }



        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO handle activation function parameter gradients

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }
    	
//	    	System.out.println("gradients: " + gradients.shapeInfoToString());
	    	
	//    	System.out.println("compute graident" + gradients);
	    	
	    	//weight gradient
	    	gradients.muli(outputweight); 
	    	
	    return gradients;
	    
	}

}
