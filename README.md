# stockheadlines
Visualization and Modeling for Stock Market/News Headlines Classification

In this analysis for my final capstone project, I evaluate the performance of dozens of models using the top 25 reddit.com/worldnews headlines over a span of eight years (2008-2016) to predict different categories measuring performance and volatility in the Dow Jones Industrial Average stock market index.  

This was an experiment within the field of Natural Language Processing and sentiment analysis on the global financial market.  The main goal was not to specifically analyze a specific word or words through positive or negative sentiment, though this was performed and will be explained later, but to analyze a longer combined string of all daily world headlines and aim to predict overall sentiment from investors based on combined current events.

The attributes from the DJIA data was as follows:
 - NetUpDown
 - Open
 - High
 - Low
 - Close
 - Volume
 - HLdifference
 - HLcat
 - OCdifference
 - OCcat
 
HLdifference describes the difference between the daily ‘High’ and ‘Low’ values.  OCdifference does the same but for ‘Open’ and ‘Close’ values.  

HLcat is a categorical variable used to describe the range of difference from HLdifference as such: 
 1 : 0 - 100 (low volatility)
 2 : 100 - 250 (medium volatility)
 3 : 250+ (high volatility)
 
 OCcat is a categorical variable used to describe the range of difference from OCdifference as such:
 2 : >100 (very good)
 1 : >0 (good)
 -1 : <0 (bad)
 -2 : <-100 (very bad)
 
The variable ‘NetUpDown’ as shown in the first table describes if the market was either up (1) or down (0) overall on the day.  These three categorical variables, NetUpDown, OCcat, and HLcat, are the variables for which I perform my classification analysis.  
 
There are slightly more positive days and negative days and more low volatility days than high. 

Notice the overall market incline that ensued following the great financial crisis in 2008/2009.  It is important to visualize this chart to understand the market climate in which this analysis was performed.
 
I split the data into train and test sets (roughly an 80/20 split) and combined all 25 headlines from each day into a single string.  

The use of CountVectorizer() within the sci-kit learn package was used to break down each headline into an array of individual words.  From there I created a matrix using fit_transform() to create a numeric array of the large string of words, similar to OneHot Encoding.  This created a value for each unique word in the array.  This initial step of breaking down the headline was vital to begin modeling.

I first began modeling using ‘NetUpDown’ as the y-variable using the LogisticRegression() algorithm.  I set up 3 models: a one-word, two-word, and three-word model.  The one-word model utilized each unique word in the entire array, a total of 31,122 unique words across all headlines.  The two-word model used two-word pairings adjacent throughout the whole array, a total of 355,342 unique two-word arrays.  The three-word model did the same thing, equaling a total of 589,589 unique three-word arrays across all headlines.  The ‘NetUpDown’ prediction accuracy using the LogisticRegression() algorithm was as follows for each model:
One-Word: 46%
Two-Word: 55.8%
Three-Word: 52.1%

Using the LogisticRegression() algorithm I was able to determine positive and negative coefficients, or word arrays that had meaningful impact on prediction results. Some positive one-word examples include: ‘olympics’ ‘votes’ and ‘turn’.  Some negative one-word examples include: ‘sanctions’ and ‘criminal’.  Looking at the two-word model a positive array was ‘the first’ and a negative one was ‘with iran’.  But looking at the three-word model is where I decided to make a change with how the headlines were being analyzed.  Some of the positive coefficients were arrays such as ‘this is the’ and ‘the right to’ while the negative ones were ‘said to be’ and ‘in the country’.  The smaller words like ‘the’ and ‘in’ did not provide any sort of description to what was being analyzed.  They were simply filler words not made for any impact in the headlines but somehow were having a large effect on the models.  This is when I looked into removing the smaller, non-essential words, also called stop words. 
 
I defined a new function through a series of for loops to remove the stop words from the long headline string I originally created.  However, the results were varied.  Examining the word coefficients and the impact of single arrays on the prediction results looked to make more sense.  For example, some positive two- and three-word arrays were ‘high court’ and ‘nobel peace prize’ while some negative ones were ‘nuclear weapons’ and ‘phone hacking scandal’ respectively.  On the surface it seemed like removing the stop words would add more clarity to the model and allow for better prediction results since the positive and negative coefficients were no longer cluttered with stop words.  However, I found my prediction results using the same LogisticRegression() algorithm on the ‘NetUpDown’ y-variable were actually worse than the previous modeling with the stop words.
One-Word: 43.9%
Two-Word: 51.6%
Three-Word: 51.6%

I continued to model results for my other categorical variables HLcat and OCcat.  These variables included more options for the model to consider, rendering the accuracy to not be as high as NetUpDown.  For example, the best accuracy when modeling for OCcat was using the Support Vector Machine (SVM) algorithm on the headlines that included stop words, and it was about 31%.  Given that a random guess on 4 categorical variables is 25%, this was a slight improvement but indicated the power of machine learning to improve accuracy.  However, that same SVM model on OCcat when used on the headlines with stop words removed only performed at 24.8% accuracy.  Almost identical accuracy as a random guess.
 
 One, Two, and Three Word Positive and Negative Coeffecient Results without stopwords using LogisticRegression() below:
 
Positive One Word	Coefficients
15518	kills	0.584569
24864	set	0.470890
18336	mumbai	0.435496
27684	territory	0.430944
18856	network	0.420127
19004	nigeria	0.414495
19697	olympics	0.414448
24716	seize	0.394041
29830	votes	0.392725
10949	first	0.391790
 
Negative One Word	Coefficients
13526	hours	-0.435569
6556	congo	-0.443189
26066	speech	-0.444393
16596	low	-0.455239
24028	run	-0.488109
7006	country	-0.512273
7164	criminal	-0.516699
24594	sea	-0.527582
3577	begin	-0.551698
24237	sanctions	-0.563508

Postive Two Word	Coefficients
121455	first time	0.346772
146840	high court	0.259975
212871	new zealand	0.256025
311310	tear gas	0.253324
279666	security council	0.224942
78301	court rules	0.213831
226324	palestinian state	0.208117
276214	says russia	0.199270
293320	south korea	0.193078

Negative Two Word	Coefficients
233007	phone hacking	-0.174392
338846	wall street	-0.186267
40759	bin laden	-0.188260
149587	hong kong	-0.191700
162677	iran nuclear	-0.195982
216164	nuclear weapons	-0.201817
283940	sexual abuse	-0.202063
293321	south korean	-0.280352
25985	around world	-0.302241

Positive Three Word	Coefficients
265355	nobel peace prize	0.250569
214896	kim jong un	0.149927
185664	human rights watch	0.143440
147652	first time since	0.135138
407043	un security council	0.130380
21187	al jazeera english	0.109584
3637	18 year old	0.109397
302203	president hosni mubarak	0.106104
302127	president evo morales	0.103716
282241	papua new guinea	0.103163

Negative Three Word	Coefficients
157506	fukushima nuclear plant	-0.095381
71333	chancellor angela merkel	-0.096266
162293	german chancellor angela	-0.101639
289945	phone hacking scandal	-0.104505
249405	missile defense system	-0.107351
382065	syrian security forces	-0.110006
38059	aung san suu	-0.112759
339226	san suu kyi	-0.112759
435053	world war ii	-0.119638
277944	osama bin laden	-0.161913
