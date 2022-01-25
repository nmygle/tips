import matplotlib.pyplot as plt
import seaborn as sns

data={'feature_names': model.feature_names_,'feature_importance': model.get_feature_importance()}
fi_df = pd.DataFrame(data)
#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
#Define size of bar plot
plt.figure(figsize=(4,8))
#Plot Searborn bar chart
sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#Add chart labels
plt.title('FEATURE IMPORTANCE')
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')
plt.show()
