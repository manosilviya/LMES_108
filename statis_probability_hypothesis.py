import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
# 📌 Data for A and B
visitors_A, conversions_A = 200, 30
visitors_B, conversions_B = 220, 20

# Conversion rates
rate_A = conversions_A / visitors_A
rate_B = conversions_B / visitors_B

print(f"Conversion Rate A: {rate_A:.3%}")
print(f"Conversion Rate B: {rate_B:.3%}")

# 🎯 Hypothesis:
# H0: No difference between A and B (conversion rates are the same)
# H1: B has a better conversion rate than A
count = np.array([conversions_A, conversions_B])
count_without_array = [conversions_A, conversions_B]
nobs = np.array([visitors_A, visitors_B])
print(count)
print(type(count))
print(count_without_array)
print(type(count_without_array))
# 🔍 Test — Two proportions Z-test
# count = np.array([conversions_A, conversions_B])
# nobs = np.array([visitors_A, visitors_B])
#
z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
#
print(f"Z-statistic: {z_stat:.2f}")
print(f"P-value: {p_value:.3f}")
#
# # 🧠 Conclusion
if p_value < 0.05:
    print("✅ Reject the null hypothesis: Version B is significantly better!")
else:
    print("❌ Fail to reject the null hypothesis: No significant difference.")
