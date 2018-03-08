package net.mydreamy.mlpharmaceutics.transfer;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossL1;

public class WeightedL1Loss extends LossL1 {
	
	private double outputweight;
	
	
	public WeightedL1Loss() {
        this(null);
    }
	
	public WeightedL1Loss(INDArray weights) {		
		super(weights);
	}
	
	public WeightedL1Loss(INDArray weights, double outputweight) {
	
		super(weights);
		
		this.outputweight = outputweight;
		
	}
	
	public WeightedL1Loss(double outputweight) {
		
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
    	
    	//invoke super computeGradient
    	INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
    	
//    	System.out.println("compute graident" + gradients);
    	
    	//weight gradient
    	gradients.muli(outputweight);
    	
        return gradients;
    }

}
