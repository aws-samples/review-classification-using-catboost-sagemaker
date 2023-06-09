## Implementing a Review classification model with Catboost and SageMaker
The solution demonstrates a Distributed Learning system implementing Catboost using bring your own architecture on Sagemaker.

#### Background
  Data set: Amazon Customer Reviews Dataset
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazon’s iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. Over 130+ million customer reviews are available to researchers as part of this dataset.

#### Approach
1.	The Review classification is NLP machine learning model to predict whether a review posted by the customer is positive or negative. For the sake of simplification, we have converted the ratings provided by the customer into a binary target variable with value equals to 1 when ratings were either 4 or 5.

2.	In present work, we split the data into train valdiation and test. Train data is used for training while validation data is used to evaluate different hyperparameters.  

3.	The data was trained using SageMaker bring your own custom Docker container. In the present example the training script consisted of following function and their implementation.
    1. Dockerfile with steps to build the docker and mounting necessary drivers for GPU enabled training
    2. Script to build the docker and push to ECR
    3. Train script to execute training
    	1. Read training data in pandas data frame
    	2. Limit categorical variables to maximum 250 levels
    	3. textual features are processed with character tri gram with maximum size of 100000
    	4. PR-AUC score to optimize the HPO job.

4.	The model was evaluated on the test data using SageMaker Batch Transform functionality.
5.	Model achieved an ROC-AUC of 0.98 on the holdout set below are some reviews classified as good and bad by the model
    ###### Reviews classified as Good by the model

           1. Great show! Brilliant dialogue! Excellent acting, superb storyline, wonderful photography !!😊
      	   2. One of the all-time BEST shows ever!  Can't wait for the next episode! Addictive!,
       	   3. Excellent!  So much fun!  Great movie!!  We LOVED it!!
           4. Fantastic Series! I love it! I'm on Season 4!!!!!!!!!!!!!!!! AAA++++++++++++++++++++++++++++++++++++++=
           5. Excellent. Awesome. Amazing. LOVE THE SHOW!!!!!!!
       
     ###### Reviews classified as Bad by the model

             1. Worst movie I have seen in a long time... Horrible acting. Poorly written. Ridiculously cheesy. Lures you in with a promise of supernatural danger that turns out to be the dumbest of the dumb. Don't waste your time.
    	     2. Not even worth watching for free.  Terrible.  Horrible.  Very Bad.  Should never have been made.  Waste of time.  Crap.  Trash.  Couldn't even finish it.
             3. Why can't I give this no stars? Absolute garbage, stupid, boring, poorly written, bad acting, and terrible direction. I won’t waste another word on this trash. I already lost a couple of hours on my life. Stare at a wall for 2 hours instead of watching this.
             4.Terrible. Poorly written.  Predictable.  Disappointed.
             5.Horrible. Slow. Boring.  What a disappointment.
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
