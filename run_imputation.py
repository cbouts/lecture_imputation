import pandas
from sklearn import linear_model

from sklearn.model_selection import KFold
from sklearn import metrics
# these things allow us to get a r2 score before we fill in the missing data. 


# loading the dataset
dataset_missing = pandas.read_csv("dataset_missing.csv", dtype='object')
# specifying "object" just makes it run faster. object basically just means everything.

print(dataset_missing)
# the dataset looks at 10000 individuals over 11 years

print(dataset_missing.isnull().sum())
# isnull finds null values in the pandas dataframe. It returns a bunch of true or false. when value is missing, you get true, when value is not missing, you get false.
# sum sums them together according to the columnss
# by printing this, we find that nothing but earnings is missing. there are 5142 missing values for earnings
# we don't want to just drop these observations. 
# if you have over 20% of data missing, this doesn't work. but here, we have 5% missing so it's not that bad. we can try to make a guess about what earnings are.
# so guessing a value for earnings: is related to ability, age, occupation, region, gender, education, exp, years.
# earning equation (Lintzer equation?) says that the y vbl is earnings. BUT here we aren't looking for an equation like this. we are trying to use what info we have to guess the correct earnings.
# we don't need to deal with endogeneity here (even though it is present here) because it's a PREDICTIVE exercise

# we are going to impute earnings. How to actually do this.

# for i in range(2000,2011):
# 	print("year: ", str(i))
# 	print(dataset_missing[dataset_missing['year']==str(i)].isnull().sum())
# 	# goes to data missing, gets all the rows where year = i. 
# 	# now we're doing this for each particular year. 
# 	# the point of this is to get a closer look at the missing data.


def data_imputation(dataset, impute_target_name, impute_data):
# writing a function. it needs the dataset that we use for the imputation, the impute target (which vbl we are going to impute), and impute_data (what data is used to impute the target)
	impute_target = dataset[impute_target_name]
	sub_dataset = pandas.concat([impute_target, impute_data], axis = 1)
	# sub_dataset is the impute target combined with the impute data.
	# concat allows us to merge 2 datasets together
	# axis = 1 means it's a horizontal concat.
	# we use the sub_dataset to get the data
	data = sub_dataset.loc[:, sub_dataset.columns != impute_target_name][sub_dataset[impute_target_name].notnull()].values
	# loc is what we use when the target is not ht
	# all the columns that are not equal to the impute_target_name
	# gets all the subdataset rows where sub_dataset
	target = dataset[impute_target_name][sub_dataset[impute_target_name].notnull()].values
	
	# form the kfold object, for how we do the testing
	kfold_object = KFold(n_splits=4)
	kfold_object.get_n_splits(data)
	i=0
	for training_index, test_index in kfold_object.split(data):
		# we loop thrugh all the cases (4) that we have. in each case:
		i=i+1
		print("case: ", str(i))
		data_training = data[training_index]
		data_test = data[test_index]
		# training_index gets corresponding rows
		target_training = target[training_index]
		target_test = target[test_index]
		# because we have multiple machines:
		one_fold_machine = linear_model.LinearRegression()
		# linear regression constructor ^
		one_fold_machine.fit(data_training, target_training)
		new_target = one_fold_machine.predict(data_test)
		# now we use whatever metrics we want
		print(metrics.mean_absolute_error(target_test, new_target))

	machine = linear_model.LinearRegression()
	machine.fit(data, target)
	return machine.predict(sub_dataset.loc[:, sub_dataset.columns != impute_target_name].values)
	# we want to get the sub_dataset and get all the collumns that are not the impute_target_name.
	# in predict we put 110,000 rows, so we get 110,000 predictions. we only really need the 5412 ones taht are missing, but this preserves the positions of each of them. 

# we have other category variables like occupation and region. it would work to put dummies in, but the distance between region 1 and region 2 is not the same as that btwn 3 and 2. for category data, need to put in dummies. how do we do that?
# forming dummies:
region_dummies = pandas.get_dummies(dataset_missing["region"])
# this thing is taking in not an array, but only one column of the dataset_missing, so we only need one []
occ_dummies = pandas.get_dummies(dataset_missing["occ"])
# to put these in dataframe, use concat.


impute_data = pandas.concat([dataset_missing[["ability","age","female","education","exp"]],region_dummies,occ_dummies], axis = 1)
# concat([stuff_you_want_to_concat], axis)
# axis = 1 you're merging horizontally instead of vertically

# using the function: call it
# impute_data = dataset_missing[["ability","age","female","education","exp"]]
# impute_data = dataset_missing[["female"]]
# putting in fewer things here [[]]  makes it less accurate
# dataset_missing[] is syntax for us to get a part of the df (a column, gotten with an index number). we can also put in the name of the column, but we need 2 [[]] because with [] it takes an array, not a string. to form an array of a string, use [] within [].
# we use [[]] to get an array with 5 strings.

new_earnings = data_imputation(dataset_missing, "earnings", impute_data)
# occupation and region are category variables so they can't be put into the regression- it woudln't make sense

# after putting the dummies in, it shrinks the errors.
# if we want to add in special effects, we can make additional dummies: has_graduated if education greater than 12 etc and then just concat that dummy too.

# new_earnings is an array. but it's easier to merge back if it's a df.

new_earnings = pandas.DataFrame(new_earnings)

# so now we have a df. but rename the column.
new_earnings.rename(columns={0:"earnings_imputed"}, inplace = True)
# could rename multiple columns with new_earnings.rename(columns={0:"earnings_imputed",1:"other_new"}, inplace = True)
# inplace = True SAVES THE CHANGES TO THE DATAFRAME new_earnings.
# alternatively, we could use new_earnings = new_earnings.rename(columns={0:"earnings_imputed"}) but this is slower.

print(new_earnings)

# now put it back in the dataset
dataset_missing = pandas.concat([dataset_missing, new_earnings], axis = 1)

dataset_missing['earnings_missing'] =dataset_missing['earnings'].isnull()
# this defines a new dummy vbl earnings_missing, which is true where earnings is null and false when earnings has already been reported.

print(dataset_missing)

# so NOW, it contains both earnings and earnings_imputed. for those that already had earnings, earnings_imputed is different/wrong
# we need to fill in earnings with earnings_imputed if earnings is missing!! and leave earnings otherwise.
print(dataset_missing.isnull().sum())
# shows how many null values we have in each column of the new dataset.

dataset_missing['earnings'].fillna(dataset_missing['earnings_imputed'], inplace=True)
# this only works if you have all 110000 predictions already.
print(dataset_missing.isnull().sum())
# and now there are 0 missing values in the earnings because they have been immputed.

# we don't have to do this, but we drop the unnecessary columns now (it makes it easier on other ppl).
dataset_missing.drop(columns=["earnings_imputed","earnings_missing"], inplace=True)

# we could round off earnings consistently because if we don't do this, the imputed values for earnings go on for a really long time. 
# we'd have to look up how to do this in pandas

dataset_missing['earnings'] = dataset_missing['earnings'].astype(float).round(2)
# this rounds the earnings to 2 decimal places instead of having 


# and now we write it to a file.
dataset_missing.to_csv("data_not_missing.csv", index=False)
# index=False turns off the thing where to_csv creates and appends an additional (unnecessary) index column. we don't need this because we already had a column called "id" with index. 
# most of the time though, we don't want to do index=false because we wouldn't already have an index.


