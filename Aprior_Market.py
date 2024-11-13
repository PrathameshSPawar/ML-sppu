import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv('/Market_Basket_Optimisation.csv', header=None)

transactions = []
for i in range(data.shape[0]):
    transaction = [str(data.values[i, j]) for j in range(data.shape[1]) if str(data.values[i, j]) != 'nan']
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)

print("Association Rules:")
print(rules)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

rules_high_confidence = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

print("Association Rules with Higher Confidence:")
print(rules_high_confidence)
