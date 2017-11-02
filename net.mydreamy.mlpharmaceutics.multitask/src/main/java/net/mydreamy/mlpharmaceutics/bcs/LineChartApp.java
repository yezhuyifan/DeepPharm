package net.mydreamy.mlpharmaceutics.bcs;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Side;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.NumberAxis;
import javafx.stage.Stage;


/**
 * A chart in which lines connect a series of data points. Useful for viewing
 * data trends over time.
 *
 * @sampleName Line Chart
 * @preview preview.png
 * @see javafx.scene.chart.LineChart
 * @see javafx.scene.chart.NumberAxis
 * @related /Charts/Area/Area Chart
 * @related /Charts/Scatter/Scatter Chart
 * @docUrl https://docs.oracle.com/javafx/2/charts/jfxpub-charts.htm Using JavaFX Charts Tutorial
 * @playground chart.data
 * @playground - (name="xAxis")
 * @playground xAxis.autoRanging
 * @playground xAxis.forceZeroInRange
 * @playground xAxis.lowerBound (min=-10,max=10,step=0.2)
 * @playground xAxis.upperBound (max=10,step=0.2)
 * @playground xAxis.tickUnit (max=3,step=0.2)
 * @playground xAxis.minorTickCount (max=16)
 * @playground xAxis.minorTickLength (max=15)
 * @playground xAxis.minorTickVisible
 * @playground xAxis.animated
 * @playground xAxis.label
 * @playground xAxis.side
 * @playground xAxis.tickLabelFill
 * @playground xAxis.tickLabelGap
 * @playground xAxis.tickLabelRotation (min=-180,max=180,step=1)
 * @playground xAxis.tickLabelsVisible
 * @playground xAxis.tickLength
 * @playground xAxis.tickMarkVisible
 * @playground - (name="yAxis")
 * @playground yAxis.autoRanging
 * @playground yAxis.forceZeroInRange
 * @playground yAxis.lowerBound (min=-5,max=5,step=0.2)
 * @playground yAxis.upperBound (max=10,step=0.2)
 * @playground yAxis.tickUnit (max=3,step=0.2)
 * @playground yAxis.minorTickCount (max=16)
 * @playground yAxis.minorTickLength (max=15)
 * @playground yAxis.minorTickVisible
 * @playground yAxis.animated
 * @playground yAxis.label
 * @playground yAxis.side
 * @playground yAxis.tickLabelFill
 * @playground yAxis.tickLabelGap
 * @playground yAxis.tickLabelRotation (min=-180,max=180,step=1)
 * @playground yAxis.tickLabelsVisible
 * @playground yAxis.tickLength
 * @playground yAxis.tickMarkVisible
 * @playground - (name="chart")
 * @playground chart.horizontalGridLinesVisible
 * @playground chart.horizontalZeroLineVisible
 * @playground chart.verticalGridLinesVisible
 * @playground chart.verticalZeroLineVisible
 * @playground chart.animated
 * @playground chart.legendSide
 * @playground chart.legendVisible
 * @playground chart.title
 * @playground chart.titleSide
 * @embedded
 */
public class LineChartApp extends Application {

    private LineChart chart;
    private NumberAxis xAxis;
    private NumberAxis yAxis;
    private Stage primaryStage;
    
    private ObservableList<XYChart.Series<Double,Double>> TrainlineChartData;
    private ObservableList<XYChart.Series<Double,Double>> DevlineChartData;
    private ObservableList<XYChart.Series<Double,Double>> TestlineChartData;

    private ObservableList<XYChart.Series<Double,Double>> TrainDevTestBiolineChartData;
    private ObservableList<XYChart.Series<Double,Double>> TrainDevTestBPBlineChartData;
    private ObservableList<XYChart.Series<Double,Double>> TrainDevTestHLlineChartData;
    private ObservableList<XYChart.Series<Double,Double>> TrainDevTestVDlineChartData;
    
    private List<double[]> MSEs;
    private int mselength;
    private List<double[]> MSEDevs;
    private int devlength;
    private List<double[]> MSETests;
    private int testlength;
    
    private ModelTrainingMultiTaskWihoutNormalizationForGUI model;

    public void getAccuaryFromModel(String resulttype) {
    	
    	if (resulttype.equals("MAE")) {
    		
           	MSEs = model.getMSEs();
        	mselength = MSEs.size();
        	
        	MSEDevs = model.getMSEDevs();
        	devlength = MSEDevs.size();
        	
        	MSETests = model.getMSETs();
        	testlength = MSETests.size();
    		
    	}
    	
    	if (resulttype.equals("MAELessTen")) {
    		
           	MSEs = model.getAccurecyMAEs();
        	mselength = MSEs.size();
        	
        	MSEDevs = model.getAccurecyMAEDevs();
        	devlength = MSEDevs.size();
        	
        	MSETests = model.getAccurecyMAETs();
        	testlength = MSETests.size();
    		
    	}
    	

    	
    }
    
    public void trainAndPrepareChart() {
    	
    	model = new ModelTrainingMultiTaskWihoutNormalizationForGUI();
    	model.train();
    	
    	getAccuaryFromModel("MAELessTen"); 
    	

        
        TrainlineChartData = FXCollections.observableArrayList();
        DevlineChartData = FXCollections.observableArrayList();
        TestlineChartData = FXCollections.observableArrayList();

        TrainDevTestBiolineChartData = FXCollections.observableArrayList();
        TrainDevTestBPBlineChartData = FXCollections.observableArrayList();
        TrainDevTestHLlineChartData = FXCollections.observableArrayList();
        TrainDevTestVDlineChartData = FXCollections.observableArrayList();

        
        //Train
        ObservableList<XYChart.Data> Biomsetrain =  FXCollections.observableArrayList();   
        for (int i = 0; i < mselength; i++) {
        	
        	double[] temp = MSEs.get(i);
        	Biomsetrain.add(new XYChart.Data<>(i, temp[0]));
        	
        }
        TrainlineChartData.add(new LineChart.Series("Bio MSE Trainning Set", Biomsetrain));
        TrainDevTestBiolineChartData.add(new LineChart.Series("Bio MSE Trainning Set", Biomsetrain));
        
        ObservableList<XYChart.Data> BPBmsetrain =  FXCollections.observableArrayList();   
        for (int i = 0; i < mselength; i++) {
        	
        	double[] temp = MSEs.get(i);
        	BPBmsetrain.add(new XYChart.Data<>(i, temp[1]));
        	
        }
        TrainlineChartData.add(new LineChart.Series("BPB MSE Trainning Set", BPBmsetrain));
        TrainDevTestBPBlineChartData.add(new LineChart.Series("BPB MSE Trainning Set", BPBmsetrain));
        
        ObservableList<XYChart.Data> HLtrain =  FXCollections.observableArrayList();   
        for (int i = 0; i < mselength; i++) {
        	
        	double[] temp = MSEs.get(i);
        	HLtrain.add(new XYChart.Data<>(i, temp[2]));
        	
        }
        TrainlineChartData.add(new LineChart.Series("HL MSE Trainning Set", HLtrain));
        TrainDevTestHLlineChartData.add(new LineChart.Series("HL MSE Trainning Set", HLtrain));
        
        ObservableList<XYChart.Data> VDtrain =  FXCollections.observableArrayList();   
        for (int i = 0; i < mselength; i++) {
        	
        	double[] temp = MSEs.get(i);
        	VDtrain.add(new XYChart.Data<>(i, temp[3]));
        	
        }
        TrainlineChartData.add(new LineChart.Series("VD MSE Trainning Set", VDtrain));
        TrainDevTestVDlineChartData.add(new LineChart.Series("VD MSE Trainning Set", VDtrain));
        
        
        //Dev
        ObservableList<XYChart.Data> BioDev=  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSEDevs.get(i);
        	BioDev.add(new XYChart.Data<>(i, temp[0]));
        	
        }
        DevlineChartData.add(new LineChart.Series("Bio MSE Validation Set", BioDev));
        TrainDevTestBiolineChartData.add(new LineChart.Series("Bio MSE Validation Set", BioDev));

    
        ObservableList<XYChart.Data> BPBDev =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSEDevs.get(i);
        	BPBDev.add(new XYChart.Data<>(i, temp[1]));
        	
        }
        DevlineChartData.add(new LineChart.Series("BPB MSE Validation Set", BPBDev));
        TrainDevTestBPBlineChartData.add(new LineChart.Series("BPB MSE Validation Set", BPBDev));
        
        ObservableList<XYChart.Data> HLDev =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSEDevs.get(i);
        	HLDev.add(new XYChart.Data<>(i, temp[2]));
        	
        }
        DevlineChartData.add(new LineChart.Series("HL MSE Validation Set", HLDev));
        TrainDevTestHLlineChartData.add(new LineChart.Series("HL MSE Validation Set", HLDev));
        
        ObservableList<XYChart.Data> VDDev =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSEDevs.get(i);
        	VDDev.add(new XYChart.Data<>(i, temp[3]));
        	
        }
        DevlineChartData.add(new LineChart.Series("VD MSE Validation Set", VDDev));
        TrainDevTestVDlineChartData.add(new LineChart.Series("VD MSE Validation Set", VDDev));
        
        //Test
        ObservableList<XYChart.Data> BioTest=  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSETests.get(i);
        	BioTest.add(new XYChart.Data<>(i, temp[0]));
        	
        }
        TestlineChartData.add(new LineChart.Series("Bio MSE Testing Set", BioTest));
        TrainDevTestBiolineChartData.add(new LineChart.Series("Bio MSE Testing Set", BioTest));

    
        ObservableList<XYChart.Data> BPBTest =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSETests.get(i);
        	BPBTest.add(new XYChart.Data<>(i, temp[1]));
        	
        }
        TestlineChartData.add(new LineChart.Series("BPB MSE Testing Set", BPBTest));
        TrainDevTestBPBlineChartData.add(new LineChart.Series("BPB MSE Testing Set", BPBTest));
        
        ObservableList<XYChart.Data> HLTest =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSETests.get(i);
        	HLTest.add(new XYChart.Data<>(i, temp[2]));
        	
        }
        TestlineChartData.add(new LineChart.Series("HL MSE Testing Set", HLTest));
        TrainDevTestHLlineChartData.add(new LineChart.Series("HL MSE Testing Set", HLTest));
        
        ObservableList<XYChart.Data> VDTest =  FXCollections.observableArrayList();   
        for (int i = 0; i < devlength; i++) {
        	
        	double[] temp = MSETests.get(i);
        	VDTest.add(new XYChart.Data<>(i, temp[3]));
        	
        }
        TestlineChartData.add(new LineChart.Series("VD MSE Testing Set", VDTest));
        TrainDevTestVDlineChartData.add(new LineChart.Series("VD MSE Testing Set", VDTest));
    	
    }
    
    public Parent createContent(ObservableList<XYChart.Series<Double,Double>> data) {
    	
        xAxis = new NumberAxis("Epoch", 0, 100, 10);
        yAxis = new NumberAxis("MSE", 0, 1, 0.1);
    	
        chart = new LineChart(xAxis, yAxis);
        chart.getData().addAll(data);
        chart.setCreateSymbols(false);
        chart.setLegendSide(Side.TOP);
        
        return chart;
        
    }


    
    
    @Override public void start(Stage primaryStage) throws Exception {
    
    	trainAndPrepareChart();
    	
        primaryStage.setScene(new Scene(createContent(TrainDevTestBiolineChartData), 1500, 800));
        primaryStage.setTitle("Bioavalibility MSE:  Train vs Dev vs Test ");
        
        this.primaryStage = primaryStage;
        primaryStage.show();
        
        Stage BPB = new Stage();
        BPB.setScene(new Scene(createContent(TrainDevTestBPBlineChartData), 1500, 800));
        BPB.setTitle("BPB MSE:  Train vs Dev vs Test ");
        BPB.show();
        
        Stage HL = new Stage();
        HL.setScene(new Scene(createContent(TrainDevTestHLlineChartData), 1500, 800));
        HL.setTitle("HL MSE:  Train vs Dev vs Test ");
        HL.show();
        
        Stage VD = new Stage();
        VD.setScene(new Scene(createContent(TrainDevTestVDlineChartData), 1500, 800));
        VD.setTitle("VD MSE:  Train vs Dev vs Test ");
        VD.show();
        
    }
    


    /**
     * Java main for when running without JavaFX launcher
     * @param args command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }

}
