package net.mydreamy.mlpharmaceutics.transfer;

public class SingularAssesmentMetrics {

	int datasetnum = 0;
	
	int truepostivenum = 0;
	int falsepositivenum = 0;
	int falsenegativenum = 0;
	int truenegativenum = 0;
	
	int postivenum = 0;
	int negativenum = 0;
	
	int truepositive = 0;
	int falsenegative = 0;
	int falsepositive = 0;
	int truenegatives = 0;
	int nonNaNdnum = 0;
	
	float precision = 0;
	float recall = 0;
	float f1 = 0;
	int numberOfBatch = 0;
	
	
	public void addFalsenegativenum(int n) {
		this.falsenegativenum += n;
	}
	
	public void addTruenegativenum(int n) {
		this.truenegativenum += n;
	}
	
	
	
	public int getFalsenegativenum() {
		return falsenegativenum;
	}

	public void setFalsenegativenum(int falsenegativenum) {
		this.falsenegativenum = falsenegativenum;
	}

	public int getTruenegativenum() {
		return truenegativenum;
	}

	public void setTruenegativenum(int truenegativenum) {
		this.truenegativenum = truenegativenum;
	}

	public void addFalsepositivenum(int n) {
		this.falsepositivenum += n;
	}
	
	
	
	public int getFalsepositivenum() {
		return falsepositivenum;
	}



	public void setFalsepositivenum(int falsepositivenum) {
		this.falsepositivenum = falsepositivenum;
	}



	public void addTruePostiveNum(int n) {
		this.truepostivenum += n;
	}
	
	public int getTruepostivenum() {
		return truepostivenum;
	}

	public void setTruepostivenum(int truepostivenum) {
		this.truepostivenum = truepostivenum;
	}

	public void addSetNum(int n) {
		this.datasetnum += n;
	}
	
	public void computeFinalScore() {
	
		
		 if ((this.truepostivenum + this.falsenegativenum) != 0)		
			 this.precision = this.truepostivenum / (float) (this.truepostivenum + this.falsepositivenum);
		 
		 if ((this.truepostivenum + this.falsenegativenum) != 0)
			 this.recall = this.truepostivenum / (float) (this.truepostivenum + this.falsenegativenum);
		
		 if ((precision + recall) != 0)
			 this.f1 = 2*precision*recall / (precision + recall);
		
	}
	
	public int getNumberOfBatch() {
		return numberOfBatch;
	}

	public void setNumberOfBatch(int numberOfBatch) {
		this.numberOfBatch = numberOfBatch;
	}

	public void addNonNaNnumber(int nonnan) {
		this.nonNaNdnum += nonnan;
	}
	
	public void addPostivenum(int postivenum) {
		this.postivenum += postivenum;
	}
	
	public void addNegativenum(int negativenum) {
		this.negativenum += negativenum;
	}
	
	
	public void addPrecision(float precision) {
		this.precision +=  precision;
	}
	
	public void addrecall(float recall) {
		this.recall += recall;
	}
	
	public void addf1(float f1) {
		this.f1 += f1;
	}
	
	
	


	public int getPostivenum() {
		return postivenum;
	}
	public void setPostivenum(int postivenum) {
		this.postivenum = postivenum;
	}
	public int getNegativenum() {
		return negativenum;
	}
	public void setNegativenum(int negativenum) {
		this.negativenum = negativenum;
	}
	public int getTruepositive() {
		return truepositive;
	}
	public void setTruepositive(int truepositive) {
		this.truepositive = truepositive;
	}
	public int getFalsenegative() {
		return falsenegative;
	}
	public void setFalsenegative(int falsenegative) {
		this.falsenegative = falsenegative;
	}
	public int getFalsepositive() {
		return falsepositive;
	}
	public void setFalsepositive(int falsepositive) {
		this.falsepositive = falsepositive;
	}
	public int getTruenegatives() {
		return truenegatives;
	}
	public void setTruenegatives(int truenegatives) {
		this.truenegatives = truenegatives;
	}
	public int getNonNaNdnum() {
		return nonNaNdnum;
	}
	public void setNonNaNdnum(int nonNaNdnum) {
		this.nonNaNdnum = nonNaNdnum;
	}
	public float getPrecision() {
		return precision;
	}
	public void setPrecision(float precision) {
		this.precision = precision;
	}
	public float getRecall() {
		return recall;
	}
	public void setRecall(float recall) {
		this.recall = recall;
	}
	public float getF1() {
		return f1;
	}
	public void setF1(float f1) {
		this.f1 = f1;
	}

	public int getDatasetnum() {
		return datasetnum;
	}

	public void setDatasetnum(int datasetnum) {
		this.datasetnum = datasetnum;
	}
	
	
	
}
