package Diandian;
import java.util.Iterator;
import java.util.Random;

import com.formdev.flatlaf.util.SystemInfo;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.pmml.jaxbbindings.True;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SMOTE;

public class Yutingting {

	public static void main(String[] args) throws Exception {
		
		 DataSource source = new DataSource("F:/Dropbox/cross_project_concurrency_bug_prediction/progrm_major_revision_GPU/program/weka/dataset/Accumulo.csv.arff");
		 DataSource target = new DataSource("F:/Dropbox/cross_project_concurrency_bug_prediction/progrm_major_revision_GPU/program/weka/dataset/Hadoop1.csv.arff");
		 Instances source_data = source.getDataSet();
		 Instances target_data = target.getDataSet(); 
		 //set class index to the last attribute
		 if(source_data.classIndex() == -1)
			 source_data.setClassIndex(source_data.numAttributes() - 1);
		 if(target_data.classIndex() == -1)
			 target_data.setClassIndex(target_data.numAttributes() - 1);
		 
		 //������shuffleһ��
		 Random rand_num = new Random(1);
		 source_data.randomize(rand_num);

		 // filter
		 //�ⲿ����ʹ�������Ƽ���WrapperSubsetEval��������attribute selection������ֻʣ��class,����attribute����ɾ����
//		 AttributeSelection AttrSelect = new AttributeSelection();
////		 RandomForest filter_RF = new RandomForest();
////		 NaiveBayes filter_NB = new NaiveBayes();
////		 BayesNet filter_BN = new BayesNet();
////		 J48 filter_DT = new J48();
////		 RandomForest filter_RF = new RandomForest();
////		 IBk filter_IBk = new IBk();
//		 ZeroR filter_zeroR = new ZeroR();
//		 WrapperSubsetEval eval = new WrapperSubsetEval();
////		 eval.setClassifier(filter_RF);
////		 eval.setClassifier(filter_NB);
////		 eval.setClassifier(filter_BN);
////		 eval.setClassifier(filter_DT);
////		 eval.setClassifier(filter_RF);
////		 eval.setClassifier(filter_IBk);
//		 eval.setClassifier(filter_zeroR);
////		 eval.setFolds(3);
//		 SelectedTag measure_selected = new SelectedTag(WrapperSubsetEval.EVAL_ACCURACY, WrapperSubsetEval.TAGS_EVALUATION);
//		 eval.setEvaluationMeasure(measure_selected);	 
//		 BestFirst search = new BestFirst();
//		 search.setDirection(new SelectedTag("Forward", BestFirst.TAGS_SELECTION));
//		 search.setLookupCacheSize(5);
//		 AttrSelect.setEvaluator(eval);
//		 AttrSelect.setSearch(search);
//		 AttrSelect.setInputFormat(source_data);
		 
		 // ����WrapperSubsetEval�ںܶ�������������ѡ��Ľ����ɾ�������е�attribute����ˣ����Ǹ���CfsubsetEval��
		 AttributeSelection AttrSelect = new AttributeSelection();
		 CfsSubsetEval eval = new CfsSubsetEval();
		 BestFirst search = new BestFirst();
		 search.setDirection(new SelectedTag("Forward", BestFirst.TAGS_SELECTION));
		 AttrSelect.setEvaluator(eval);
		 AttrSelect.setSearch(search);
		 AttrSelect.setInputFormat(source_data);
		 //generate new data
		 Instances source_new = Filter.useFilter(source_data, AttrSelect);
		 Instances target_new = Filter.useFilter(target_data, AttrSelect);
		 

		 //smote
		 SMOTE filter_smote = new SMOTE();
		 AttributeStats class_stats = source_data.attributeStats(source_data.numAttributes() - 1);
		 int[] class_num  = class_stats.nominalCounts;
		 int min_num;
		 int maj_num;
		 if (class_num[1] > class_num[0]) {
			min_num = class_num[0];
			maj_num = class_num[1];
		}else {
			min_num = class_num[1];
			maj_num = class_num[0];
		}
		 double smote_percentage =(double) (maj_num - min_num)*100 / (double) min_num;
		 filter_smote.setPercentage(smote_percentage);
		 filter_smote.setInputFormat(source_new);
		 Instances source_new2 = Filter.useFilter(source_new, filter_smote);

		 
		// classifier
		 RandomForest RF_final_used = new RandomForest();
		 RF_final_used.buildClassifier(source_new2);
		 // evaluate classifier and print some statistics
		 Evaluation RF_eval = new Evaluation(source_new2); 
		 eval_calculation(RF_eval, RF_final_used, target_new);
		 
		 
		 BayesNet BN_final_used = new BayesNet();
		 BN_final_used.buildClassifier(source_new2);
		 // evaluate classifier and print some statistics
		 Evaluation BN_eval = new Evaluation(source_new2); 
		 eval_calculation(BN_eval, BN_final_used, target_new);
		 
		 J48 DT_final_used = new J48();
		 DT_final_used.buildClassifier(source_new2);
		 // evaluate classifier and print some statistics
		 Evaluation DT_eval = new Evaluation(source_new2); 
		 eval_calculation(DT_eval, DT_final_used, target_new);
		 
		 Logistic LR_final_used = new Logistic();
		 LR_final_used.buildClassifier(source_new2);
		 // evaluate classifier and print some statistics
		 Evaluation LR_eval = new Evaluation(source_new2); 
		 eval_calculation(LR_eval, LR_final_used, target_new);

	}

	public static void eval_calculation (Evaluation my_eval, Classifier my_classifier, Instances my_target) throws Exception {
		my_eval.evaluateModel(my_classifier, my_target);
		String my_ConMatrix = my_eval.toMatrixString();
		String [] string_split = my_ConMatrix.split("  +| \\| +|\n +");
		 
		int TN;
		int FP;
		int FN;
		int TP;
		 if (string_split[6].equals("a = 0")) {
			System.out.println("this is diandian1");
			TN = Integer.parseInt(string_split[4]);
			FP = Integer.parseInt(string_split[5]);
			FN = Integer.parseInt(string_split[7]);
			TP = Integer.parseInt(string_split[8]);
		}else if (string_split[6].equals("a = 1")) {
			System.out.println("this is diandian2");
			TP = Integer.parseInt(string_split[4]);
			FN = Integer.parseInt(string_split[5]);
			FP = Integer.parseInt(string_split[7]);
			TN = Integer.parseInt(string_split[8]);
		}else if (string_split[5].equals("a = 0")){
			System.out.println("this is diandian3");
			String[] string_further_split = string_split[4].split(" ");
			TN = Integer.parseInt(string_further_split[0]);
			FP = Integer.parseInt(string_further_split[1]);
			FN = Integer.parseInt(string_split[6]);
			TP = Integer.parseInt(string_split[7]);
		}else if (string_split[5].equals("a = 1")) {
			System.out.println("this is diandian4");
			String[] string_further_split = string_split[4].split(" ");
			TP = Integer.parseInt(string_further_split[0]);
			FN = Integer.parseInt(string_further_split[1]);
			FP = Integer.parseInt(string_split[6]);
			TN = Integer.parseInt(string_split[7]);
		}else {
			TN = 0;
			FP = 0;
			FN = 0;
			TP = 0;
		}
		 
		 
		 double PD = (double) TP / (double) (TP+FN);
		 double PF = (double) FP / (double) (FP+TN);
		 double Bal = 1 - Math.sqrt((Math.pow(PF, 2) + Math.pow((1-PD), 2))/2);
		 double AUC = my_eval.areaUnderROC(1);
		 double Precision = my_eval.precision(1);
		 double Recall = my_eval.recall(1);
		 double F_measure = my_eval.fMeasure(1);
		 System.out.println("PD: " + PD);
		 System.out.println("PF: " + PF);
		 System.out.println("Bal: " + Bal);
		 System.out.println("AUC: " + AUC);
		 System.out.println("Precision: " + Precision);
		 System.out.println("Recall: " + Recall);
		 System.out.println("F_measure: " + F_measure + "\n");

	}
}
